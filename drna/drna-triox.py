import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
D‑RNA: Dual‑Helix Resonance Neural Architecture (DRNA) Pre-Norm･Kv-RoPE 版
仕様：Pre-Norm(RMSNorm)、GELU(Activation)、Kv-RoPE(head_dim)、mask(padding + causal)
Transformerの全接続性を継承しつつ、二重らせん(Dual-Helix)構造による
｢共鳴収縮｣(Resonant Contraction)を物理的に再現したニューラルアーキテクチャです
螺旋の同期：Attention(文脈の回想)とMLP(知識の定着)を並列配置し RoPE で情報を同期
位相の保持：RoPE(Phase Field)を回転場として利用し、安定した相対位置を保ち早期収束を両立
高密度圧縮：Pre-Norm により、各らせんを安定的に収縮させ、全結合により記憶を定着させる
'''

# 3値誘導制御コア(モデルの書き換え、3値ブレンド、ペナルティ計算、結晶化)
class TernaryTrainingManager:
    """
    D-RNAのコードに触れることなく、3値誘導の全ライフサイクルを統括する抽象化マネージャー。
    """
    def __init__(self, model, warmup_steps=100, max_lambda=1.0):
        self.model = model
        self.warmup_steps = warmup_steps
        self.max_lambda = max_lambda
        self.current_step = 0
        self.total_steps = 0
        
        # 内部で利用するステッププロバイダー関数
        def step_provider():
            return self.current_step, self.total_steps
            
        # 1. モデル内の全2次元重みにフックを外付け
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # ★ 拡張性：LLMの生命線である「embed」と「output_head」は絶対に3値化しない
                if "embed" in name or "output_head" in name:
                    continue
                if not hasattr(module, "raw_weight"):
                    # 元の実数重みを聖域(raw_weight)に退避
                    module.register_parameter("raw_weight", nn.Parameter(module.weight.data.clone()))
                    delattr(module, "weight")
                    module.register_buffer("weight", module.raw_weight.data.clone())
                module.register_forward_pre_hook(TernaryWeightHook(step_provider, self.warmup_steps))

    def amend_loss(self, task_loss, step, total_steps):
        """
        【ループ内抽象化用】
        メインのタスク損失（CrossEntropy等）を受け取り、
        現在のステップに応じた3値結晶化ペナルティを自動計算して合算した損失を返します。
        """
        self.current_step = step
        self.total_steps = total_steps
        
        blend_ratio = get_ternary_schedule(step, total_steps, self.warmup_steps)
        current_lambda = blend_ratio * self.max_lambda
        
        if current_lambda == 0.0:
            return task_loss
            
        # 3値誘導トリプレット・ペナルティ自動計算
        ternary_penalty = 0.0
        for name, param in self.model.named_parameters():
            # 2次元以上の重み（LinearやEmbeddingの実数聖域）のみを対象とする
            if "raw_weight" in name and param.dim() >= 2:
                # W * (W - 1) * (W + 1) = W^3 - W を0に近づける（-1, 0, 1への収束強制力）
                ternary_penalty += torch.mean(param * (param - 1.0) * (param + 1.0)) ** 2
                
        return task_loss + current_lambda * ternary_penalty

    def export_ternary(self):
        """学習終了後、モデルの全2次元重みを完全な[-1.0, 0.0, 1.0]へ固定（結晶化）する"""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and hasattr(module, "raw_weight"):
                    param = module.raw_weight
                    dist_neg1 = torch.abs(param + 1.0)
                    dist_zero = torch.abs(param)
                    dist_pos1 = torch.abs(param - 1.0)
                        
                    stacked_dists = torch.stack([dist_neg1, dist_zero, dist_pos1], dim=-1)
                    chosen_slot = torch.argmin(stacked_dists, dim=-1)
                    
                    ternary_weight = chosen_slot.float() - 1.0
                    
                    # 最終的に実数パラメータ自体を3値に上書きして完全固定
                    module.raw_weight.copy_(ternary_weight)
                    module.weight.copy_(ternary_weight)
        return self.model

# 3値誘導外付けフックシステム(2次元重みを3値ブレンド）
def get_ternary_schedule(step, total_steps, warmup_steps=100):
    """逆転コサインアニーリングスケジューラ"""
    if step < warmup_steps:
        return 0.0
    anneal_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / anneal_steps
    return 1.0 - (0.5 * (1.0 + math.cos(progress * math.pi)))

def get_soft_ternary_weight(param, step, total_steps, warmup_steps=100):
    """勾配直通バイパス型 3値ブレンド関数"""
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
            # バックアップした実数重み(raw_weight)から疑似3値重みを計算し、一時的に上書き
            module.weight.data = get_soft_ternary_weight(module.raw_weight, step, total_steps, self.warmup_steps)

def apply_trio_induction(model, step_provider, warmup_steps=100):
    """モデルを汚さずに外側から3値化プラグインを刺す関数"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # ★ 拡張性：LLMの生命線である「embed」と「output_head」は絶対に3値化しない
            if "embed" in name or "output_head" in name:
                continue
            if not hasattr(module, "raw_weight"):
                # 元の実数重みを聖域（raw_weight）に退避
                module.register_parameter("raw_weight", nn.Parameter(module.weight.data.clone()))
                delattr(module, "weight")
                module.register_buffer("weight", module.raw_weight.data.clone())
            module.register_forward_pre_hook(TernaryWeightHook(step_provider, warmup_steps))

# D-RNA 3値化のパニックを100%いなすコンサルタント(RMSNorm 修正)
class RMSNorm(nn.Module):
    """【修正版】3値化の歪みをリセットする中心化・標準化型防波堤"""
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        # 3値化の歪みによってズレた「音量の軸」を、フル精度の実数で強制的に中心に戻すバイアス
        self.bias = nn.Parameter(torch.zeros(d_model))
        # 3値化のせいで極端にインフレ・デフレした次元ごとの音量を、個別にジャストフィットさせる実数スケール
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 1. 各次元ごとの平均と分散を計算
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        
        # 2. 3値化による「歪み・偏り」を、ここで完全にリセット（標準化）する
        x_normed = (x - mean) * torch.rsqrt(var + self.eps)
        
        # 3. リセットされた綺麗な状態に対して、フル精度のオプティマイザが最適なスケールとオフセットを施す
        return self.weight * x_normed + self.bias

class DRNA_RoPE(nn.Module):
    """二重らせんの位相を決定する回転場"""
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
    """Kによる動的位相変調済み cos/sin を受け取る"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class DRNA_Block(nn.Module):
    """DRNA共鳴ブロック：安定性を高めたPre-Norm直列共鳴構造"""
    def __init__(self, d_model, n_heads, head_dim, d_ff=None, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        # らせんA: 回想系 (Attention)
        self.norm1 = RMSNorm(d_model) # 演算の前に配置
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        # らせんB: 記憶系 (MLP)
        self.norm2 = RMSNorm(d_model) # 演算の前に配置
        d_ff = d_ff or d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), # VRAM抑制は ReLU (別レイヤの干渉で0勾配にならない｢可能性｣あり)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
            b, s, d = x.shape

            # 1. 共通の残差(ベースとなる螺旋の軸)
            residual = x

            # 2. らせんA (Attention) 並列方式
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

            # 3. らせんB (MLP)
            x_norm2 = self.norm2(x)
            m_out = self.mlp(x_norm2)

            # 4. 並列方式
            x = residual + self.dropout(a_out) + self.dropout(m_out)

            return x

class DRNA_Model(nn.Module):
    """汎用 DRNA モデルコンテナ(安定化 Pre-Norm 版)"""
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

'''
260528：３値モデル(-、0、+)学習対応／活用例"D-RNA-Trio"版を追加
260520：maskの微調整(AMP対応)／MoE-LoRA版、vlayer版、D-RNAの活用例を汎用コード化
'''

'''
汎用型 D-RNA (Pre-Norm) License: Apache License 2.0 https://github.com/muooon/DRNA
Attention is all you need_started, Resonance is all you need_endure, 
Neocognitron ― Transformer ― D‑RNA Dream Resonance Never Adjourns — it goes on...
'''
