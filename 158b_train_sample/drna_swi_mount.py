import torch
import math
from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F

'''
このコードは純粋３値学習を行うためのモデルコード(定義)です、通常はこちらを参照し呼び出すだけでOKです
既存の重みをもつモデルを読み込んで３値へ転換も可能です(モデルの次元などは学習コード側で適切に合わせてください)
---
# チェックポイント読み込み(D-RNA-Trio の学習済みモデル) ／ fp8/16/32 相当の連続重みの復元
2〜3 レイヤ：fp8相当、4〜6 レイヤ：bf16相当、7+ レイヤ：fp16/32 相当
固有位相(K-energy × RoPE 周波数成分)を重ね合わせ＋正規化(二重らせんの共鳴収縮により歪みを抑制)
特に K-for-phase の｢逐次カスケード的変化｣が位相シフトとなり自然な凸凹パターンを生成
---
D‑RNA: Dual‑Helix Resonance Neural Architecture (DRNA) Pre-Norm･Kv-RoPE 版
仕様：Pre-Norm(RMSNorm)、GELU(Activation)、Kv-RoPE(head_dim)、mask(padding + causal)
Transformerの全接続性を継承しつつ、二重らせん(Dual-Helix)構造による
｢共鳴収縮｣(Resonant Contraction)を物理的に再現したニューラルアーキテクチャです
D-RNA の位相設計と Trio Induction system により３値学習を STE に頼らず安定的に行えます
---
これは STE で機能しない optimiser などを３値学習へ活用できるようになります
将来的に３値モデルを位相差の重ねによる疑似重みをつくり、これを学習対象にして３値学習もおこなえるはずです
つまり学習元も３値モデルにできるはずです、推論も学習も３値で済むようになる最初の１歩です
'''

#---restore mount START---
class DRNAWeightRestorer:
    '''
    D-RNA の固有位相を利用した fp8/16/32 重み復元器
    チェックポイント読み込み → 位相抽出 → 重ね合わせ → 正規化 で連続分布を生成
    '''

    def __init__(self, d_model: int = 256, num_layers: int = 16):
        self.d_model = d_model
        self.num_layers = num_layers

    # チェックポイントから -/0/+ 重みを読み出す
    def load_ternary_weights(self, checkpoint_path: str) -> List[torch.Tensor]:
        '''ロードされたチェックポイントから各レイヤの重みを抽出する'''
        checkpoint = torch.load(checkpoint_path, map_location='cuda')

        layer_weights = []
        for i in range(self.num_layers):
            # QKV レイヤー(norm1 ブランチ)
            qkv_weight = checkpoint[f'layers.{i}.qkv.weight']

            # MLP 入力/出力レイヤー(norm2 ブランチ)
            mlp_0_weight = checkpoint[f'layers.{i}.mlp.0.weight']
            mlp_3_weight = checkpoint[f'layers.{i}.mlp.3.weight']

            layer_weights.extend([qkv_weight, mlp_0_weight, mlp_3_weight])

        return layer_weights

    # K-for-phase のカスケード効果から各レイヤの固有位相を抽出する
    def extract_phase_offsets(self, max_seq_len: int = 256) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        '''DRNA の固有位相(K-energy × RoPE 周波数成分)を計算し返す'''

        num_layer_groups = self.num_layers * 3

        phase_data_list = []
        base_energy = torch.arange(128, dtype=torch.float32, device='cuda') / self.d_model

        for i in range(num_layer_groups):
            # K-energy のカスケード効果：レイヤごとに滑らかにシフト(K-for-phase 同様に)
            k_energy = base_energy * (i % self.num_layers + 1)

            # DRNA の動的位相変調: tanh(k_energy) × π
            rt_phase = math.tanh(k_energy) * math.pi

            # RoPE 的な周波数成分の抽出(各レイヤ固有の周波数パターン)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, 128, 2).float() / self.d_model))

            freqs = torch.einsum("i,j->ij", 
                               torch.arange(max_seq_len, device='cuda'),
                               inv_freq)

            # cos / sin の位相シフトを生成
            cos_shift = torch.cos(freqs * rt_phase.unsqueeze(-1))
            sin_shift = torch.sin(freqs * rt_phase.unsqueeze(-1))

            phase_data_list.append((cos_shift, sin_shift))

        return phase_data_list

    # 固有位相を用いて重ね合わせ(RoPE 的な位相変調を適用)
    def superimpose_weights(
        self, 
        layer_weights: List[torch.Tensor], 
        phase_offsets: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        '''各レイヤの -/0/+ 重みに固有位相を適用し重ね合わせる'''

        reconstructed = torch.zeros_like(layer_weights[0])

        for i, w_i in enumerate(layer_weights):
            cos_shift, sin_shift = phase_offsets[i]

            # DRNA の K-for-phase 同様の位相変調: d_cos / d_sin を用いる重ね合わせ
            contribution = w_i * (cos_shift - rotate_half(w_i) * sin_shift)
            reconstructed += contribution

        return reconstructed

    # RMSNorm 的な正規化(3 値歪みのリセット)
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        '''重ね合わせ後の分布を RMSNorm で安定化する'''

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        # 中心化･標準化(3 値歪みのリセット)
        return (x - mean) * torch.rsqrt(var + 1e-8)

    # 完全復元パイプライン
    def restore_from_checkpoint(
        self, 
        checkpoint_path: str, 
        max_seq_len: int = 256
    ) -> torch.Tensor:
        '''チェックポイントを読み込み、fp8/16 相当の連続重みを生成する'''

        # -1.0/0.0/1.0 重みの読み出し
        layer_weights = self.load_ternary_weights(checkpoint_path)

        # 固有位相(K-energy × RoPE)の抽出
        phase_offsets = self.extract_phase_offsets(max_seq_len)

        # 重ね合わせ(RoPE 的な位相変調を適用)
        continuous_weight = self.superimpose_weights(
            layer_weights, 
            phase_offsets)

        # RMSNorm 的な正規化
        normalized_weight = self.normalize(continuous_weight)

        return normalized_weight

# 補助関数: DRNA rotate_half 再現
def rotate_half(x):
    '''D-RNA の K-for-phase 同様の回転操作(位相変調の右辺項)'''
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
#---restore mount END---

#---drna-swi-triox START---
# 3値誘導制御コア(モデルの書き換え、3値ブレンド、ペナルティ計算、結晶化)
class TernaryTrainingManager:
    '''
    D-RNAのコードに触れることなく、3値誘導の全ライフサイクルを統括する抽象化マネージャー
    '''
    def __init__(self, model, warmup_steps=100, max_lambda=1.0):
        self.model = model
        self.warmup_steps = warmup_steps
        self.max_lambda = max_lambda
        self.current_step = 0
        self.total_steps = 0
        
        # 内部で利用するステッププロバイダー関数
        def step_provider():
            return self.current_step, self.total_steps

        # モデル内の全2次元重みにフックを外付け
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # ★ 拡張性：LLMの生命線である｢embed｣と｢output_head｣は絶対に3値化しない
                if "embed" in name or "output_head" in name:
                    continue
                if not hasattr(module, "raw_weight"):
                    # 元の実数重みを聖域(raw_weight)に退避
                    module.register_parameter("raw_weight", nn.Parameter(module.weight.data.clone()))
                    delattr(module, "weight")
                    module.register_buffer("weight", module.raw_weight.data.clone())
                module.register_forward_pre_hook(TernaryWeightHook(step_provider, self.warmup_steps))

    def amend_loss(self, task_loss, step, total_steps):
        '''
        【ループ内抽象化用】 メインのタスク損失(CrossEntropy等)を受け取り、
        現在のステップに応じた3値結晶化ペナルティを自動計算して合算した損失を返す
        '''
        self.current_step = step
        self.total_steps = total_steps

        blend_ratio = get_ternary_schedule(step, total_steps, self.warmup_steps)
        current_lambda = blend_ratio * self.max_lambda

        if current_lambda == 0.0:
            return task_loss

        # 3値誘導トリプレット･ペナルティ自動計算
        ternary_penalty = 0.0
        for name, param in self.model.named_parameters():
            # 2次元以上の重み(LinearやEmbeddingの実数聖域)のみを対象とする
            if "raw_weight" in name and param.dim() >= 2:
                # W * (W - 1) * (W + 1) = W^3 - W を0に近づける(-1, 0, 1への収束強制力)
                ternary_penalty += torch.mean(param * (param - 1.0) * (param + 1.0)) ** 2

        return task_loss + current_lambda * ternary_penalty

    def export_ternary(self):
        '''学習終了後、モデルの全2次元重みを完全な[-1.0, 0.0, 1.0]へ固定(結晶化)する'''
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    # 拡張性：LLMの生命線である｢embed｣と｢output_head｣は絶対に3値化しない
                    if "embed" in name or "output_head" in name:
                        continue
                    # raw_weight があればそっちを本体として使う
                    if hasattr(module, "raw_weight"):
                        param = module.raw_weight
                    else:
                        param = module.weight
                    # 学習中と同じ写像：tanh(3w) で soft 3値ターゲットを作る
                    soft = torch.tanh(param * 3.0)
                    # soft を hard 3値に潰す
                    ternary = torch.zeros_like(soft)
                    ternary[soft >  0.08] =  1.0
                    ternary[soft < -0.08] = -1.0
                    # 実数パラメータ自体を3値に上書きして完全固定
                    if hasattr(module, "raw_weight"):
                        module.raw_weight.copy_(ternary)
                    module.weight.copy_(ternary)

        return self.model

# 3値誘導外付けフックシステム(2次元重みを3値ブレンド)
def get_ternary_schedule(step, total_steps, warmup_steps=100):
    '''逆転コサインアニーリングスケジューラ'''
    if step < warmup_steps:
        return 0.0
    anneal_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / anneal_steps
    return 1.0 - (0.5 * (1.0 + math.cos(progress * math.pi)))

def get_soft_ternary_weight(param, step, total_steps, warmup_steps=100):
    '''勾配直通バイパス型 3値ブレンド関数'''
    if param.dim() < 2:  # 1次元パラメータは保護
        return param
    blend_ratio = get_ternary_schedule(step, total_steps, warmup_steps)
    if blend_ratio == 0.0:
        return param
    with torch.no_grad():
        ternary_target = torch.tanh(param * 3.0)
    # 【最重要】生の勾配を裏に直通させるバイパス構造
    return param + blend_ratio * (ternary_target - param)

class TernaryWeightHook:
    def __init__(self, step_provider, warmup_steps=100):
        self.step_provider = step_provider
        self.warmup_steps = warmup_steps

    def __call__(self, module, inputs):
        step, total_steps = self.step_provider()
        if step is not None and total_steps is not None:
            # バックアップした実数重み(raw_weight)から疑似3値重みを計算し一時的に上書き
            module.weight.data = get_soft_ternary_weight(module.raw_weight, step, total_steps, self.warmup_steps)

def apply_trio_induction(model, step_provider, warmup_steps=100):
    '''モデル側ではなく外側から3値化プラグインを刺す関数'''
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # ★ 拡張性：LLMの生命線である｢embed｣と｢output_head｣は絶対に3値化しない
            if "embed" in name or "output_head" in name:
                continue
            if not hasattr(module, "raw_weight"):
                # 元の実数重みを聖域(raw_weight)に退避
                module.register_parameter("raw_weight", nn.Parameter(module.weight.data.clone()))
                delattr(module, "weight")
                module.register_buffer("weight", module.raw_weight.data.clone())
            module.register_forward_pre_hook(TernaryWeightHook(step_provider, warmup_steps))

# D-RNA 3値化のパニックを100%いなすコンサルタント(RMSNorm 修正)
class RMSNorm(nn.Module):
    '''【修正版】3値化の歪みをリセットする中心化･標準化型防波堤'''
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        # 3値化の歪みによってズレた｢音量の軸｣を、フル精度の実数で強制的に中心に戻すバイアス
        self.bias = nn.Parameter(torch.zeros(d_model))
        # 3値化のせいで極端にインフレ･デフレした次元ごとの音量を、個別にジャストフィットさせる実数スケール
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 各次元ごとの平均と分散を計算
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        # 3値化による｢歪み･偏り｣を、ここで完全にリセット(標準化)する
        x_normed = (x - mean) * torch.rsqrt(var + self.eps)

        # リセットされた綺麗な状態に対して、フル精度のオプティマイザが最適なスケールとオフセットを施す
        return self.weight * x_normed + self.bias

class DRNA_RoPE(nn.Module):
    '''二重らせんの位相を決定する回転場'''
    def __init__(self, head_dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def apply_drna_rope(q, k, cos, sin):
    '''Kによる動的位相変調済み cos/sin を受け取る'''
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class DRNA_Block(nn.Module):
    '''DRNA共鳴ブロック：安定性を高めたPre-Norm直列共鳴構造'''
    def __init__(self, d_model, n_heads, head_dim, d_ff=None, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        # らせんA：回想系 (Attention)
        self.norm1 = RMSNorm(d_model) # 演算の前に配置
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        # らせんB：記憶系 (MLP)
        self.norm2 = RMSNorm(d_model) # 演算の前に配置

        # SwiGLUの定石：パラメータ量をGELU版(×4倍)と合わせるため、約2.67倍(8/3)にする
        if d_ff is None:
            d_ff = int(2 * (d_model * 4) / 3) 

        # SwiGLUは入力に対して2つのLinear(Gate･Up)を並列に走らせます
        self.w1 = nn.Linear(d_model, d_ff, bias=False) # ゲート用
        self.w3 = nn.Linear(d_model, d_ff, bias=False) # 値用(アッププロジェクション)
        self.w2 = nn.Linear(d_ff, d_model, bias=False) # ダウンプロジェクション

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        b, s, d = x.shape

        # 共通の残差(ベースとなる螺旋の軸)
        residual = x

        # らせんA (Attention) 並列方式
        x_norm1 = self.norm1(x)

        # QKV生成 (3倍のまま)
        # ※ self.qkv(x_norm) が x_norm1 になっているか確認してください
        qkv = self.qkv(x_norm1).reshape(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # K因果的動的回転：一つ前の単語の K が今の単語の座標を決める
        # 自分の情報で自分を回さないよう、Kを1つ未来にシフトさせる
        # これにより｢赤い(K)｣が｢猫(Q,K)｣の位相を決定する構造になる
        k_for_phase = torch.cat([torch.zeros_like(k[:, :, :1, :]), k[:, :, :-1, :]], dim=2)

        # Kのエネルギーを位相(回転角)に変換
        rt_phase = torch.tanh(k_for_phase) * math.pi

        # 静的RoPE (cos, sin) を動的位相 (rt_phase) で加法定理により変調
        # ※ apply_drna_rope に rt_phase を渡せるように関数側を調整するか、
        # ここで dynamic_cos / sin を作って渡します
        d_cos = (cos * torch.cos(rt_phase)) - (sin * torch.sin(rt_phase))
        d_sin = (sin * torch.cos(rt_phase)) + (cos * torch.sin(rt_phase))

        # ２重らせんをつくる (変調された座標で回転)
        q, k = apply_drna_rope(q, k, d_cos, d_sin)

        # Attention計算
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        a_out_raw = (attn @ v).transpose(1, 2).reshape(b, s, d)
        a_out = self.out_proj(a_out_raw)

        # らせんB (MLP)
        x_norm2 = self.norm2(x)

        # SwiGLUのコア数式: Swish(xW1) * xW3(F.silu：PyTorchのSwish関数)
        gate = F.silu(self.w1(x_norm2)) 
        current_value = self.w3(x_norm2)

        # 要素積(共鳴収縮の表現としても非常に相性が良いです)
        swiglu_out = gate * current_value 

        # 最終投影
        m_out = self.w2(swiglu_out)

        # 並列方式
        x = residual + self.dropout(a_out) + self.dropout(m_out)

        return x

class DRNA_Model(nn.Module):
    '''汎用 DRNA モデルコンテナ(安定化 Pre-Norm 版)'''
    def __init__(self, vocab_size, d_model=256, n_layers=16, n_heads=8, d_ff=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head_dim = d_model // n_heads
        self.rope = DRNA_RoPE(self.head_dim)

        self.layers = nn.ModuleList([
            DRNA_Block(d_model, n_heads, self.head_dim, d_ff) for _ in range(n_layers)
        ])

        # Pre-Norm構造の場合、最終レイヤーの後に全体のNormを置くのが一般的
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None, pad_id=None):
        b, s = x.shape
        device = x.device
        inputs = x
        x = self.embed(x)

        if mask is None or mask.sum() == 0:
            # pad_id が整数(int/long)として有効な場合のみ pad_mask を作成
            # 退避させた inputs を使ってパディングを判定し因果マスクを準備
            p_id = pad_id.item() if isinstance(pad_id, torch.Tensor) else pad_id
            pad_mask = (inputs != p_id).unsqueeze(1).unsqueeze(2) if isinstance(p_id, (int, float)) else torch.ones((1, 1, 1, s), device=device, dtype=torch.bool)
            causal = torch.triu(torch.ones(s, s, device=device), diagonal=1).bool().unsqueeze(0).unsqueeze(0)

            # 環境(fp16/32)に応じた最小値を安全に自動計算
            inf_value = torch.finfo(x.dtype).min if x.dtype != torch.float16 else -65500.0

            # ゼロ初期化テンソルに無効領域をインプレースで直接埋める(カンニングの完全遮断)
            mask = torch.zeros((b, 1, s, s), device=device, dtype=x.dtype).masked_fill_(causal | (~pad_mask), inf_value)

        cos, sin = self.rope(x, x.size(1))

        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask)

        x = self.final_norm(x) # 出力前の最終同期
        return self.output_head(x)
#---drna-swi-triox END---

'''
汎用型 D-RNA (Pre-Norm) License: Apache License 2.0 https://github.com/muooon/DRNA
Attention is all you need_started, Resonance is all you need_endure, 
Neocognitron ― Transformer ― D‑RNA Dream Resonance Never Adjourns — it goes on...
'''
