import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
D‑RNA: Dual‑Helix Resonance Neural Architecture (DRNA) Pre-Norm･Kv-RoPE 版
仕様：Pre-Norm(RMSNorm)、SwiGLU(Activation)、Kv-RoPE(head_dim)、
Hybrid Layer(FFN / Vanilla)、mask(padding + causal)
Transformerの全接続性を継承しつつ、二重らせん(Dual-Helix)構造による
｢共鳴収縮｣(Resonant Contraction)を物理的に再現したニューラルアーキテクチャです
螺旋の同期：Attention(文脈の回想)とMLP(知識の定着)を並列配置し RoPE で情報を同期
位相の保持：RoPE(Phase Field)を回転場として利用し、安定した相対位置を保ち早期収束を両立
高密度圧縮：Pre-Norm により、各らせんを安定的に収縮させ、全結合により記憶を定着させる
ハイブリッド層同期：Attention(文脈回想)と回転場を共有するMLP単体層(知識定着)を交互配置(計算負荷軽減)
'''

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 2乗平均の平方根で割る(平均を減算しない･中心化をしない)
        norm = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normed

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
    """DRNA共鳴ブロック：Attentionの有無を切り替え可能な統合共鳴構造"""
    def __init__(self, d_model, n_heads, head_dim, d_ff=None, dropout=0.1, has_attn=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.has_attn = has_attn

        # らせんA：回想系 (Attention) - has_attn が True の時だけパラメータを持つ
        if self.has_attn:
            self.norm1 = RMSNorm(d_model)
            self.qkv = nn.Linear(d_model, d_model * 3)
            self.out_proj = nn.Linear(d_model, d_model)

        # らせんB：記憶系 (MLP) - 常に存在
        self.norm2 = RMSNorm(d_model)

        if d_ff is None:
            d_ff = int(2 * (d_model * 4) / 3)
            
        self.w1 = nn.Linear(d_model, d_ff, bias=False) # ゲート用
        self.w3 = nn.Linear(d_model, d_ff, bias=False) # 値用（アッププロジェクション）
        self.w2 = nn.Linear(d_ff, d_model, bias=False) # ダウンプロジェクション

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        b, s, d = x.shape
        residual = x

        # らせんA (Attention) の処理 [has_attn が True の時のみ実行]
        if self.has_attn:
            x_norm1 = self.norm1(x)
            qkv = self.qkv(x_norm1).reshape(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            k_for_phase = torch.cat([torch.zeros_like(k[:, :, :1, :]), k[:, :, :-1, :]], dim=2)
            rt_phase = torch.tanh(k_for_phase) * math.pi

            d_cos = (cos * torch.cos(rt_phase)) - (sin * torch.sin(rt_phase))
            d_sin = (sin * torch.cos(rt_phase)) + (cos * torch.sin(rt_phase))

            q, k = apply_drna_rope(q, k, d_cos, d_sin)

            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if mask is not None:
                attn = attn + mask

            attn = F.softmax(attn, dim=-1)
            a_out_raw = (attn @ v).transpose(1, 2).reshape(b, s, d)
            a_out = self.dropout(self.out_proj(a_out_raw))
        else:
            a_out = 0.0  # Attentionなし層では出力ゼロ（スキップ）

        # らせんB (MLP) の処理 [常に実行]
        x_norm2 = self.norm2(x)

        gate = F.silu(self.w1(x_norm2))
        current_value = self.w3(x_norm2)
        swiglu_out = gate * current_value
        m_out_raw = self.w2(swiglu_out)

        # 【重要】もし Attentionなし層（MLPのみ）の場合、
        # 共有されたグローバル回転場（cos, sin）をここで適用して位相の連続性を担保する
        if not self.has_attn:
            m_reshaped = m_out_raw.reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
            
            def rotate_half(tensor):
                x1, x2 = tensor.chunk(2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)
            
            m_rotated = (m_reshaped * cos) + (rotate_half(m_reshaped) * sin)
            m_out_raw = m_rotated.transpose(1, 2).reshape(b, s, d)

        m_out = self.dropout(m_out_raw)

        # 最終的な残差結合
        x = residual + a_out + m_out

        return x

class DRNA_Model(nn.Module):
    """汎用 DRNA モデルコンテナ(安定化 Pre-Norm 版・Attention有無混在対応)"""
    def __init__(self, vocab_size, d_model=256, n_layers=16, n_heads=8, d_ff=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head_dim = d_model // n_heads
        self.rope = DRNA_RoPE(self.head_dim)

        # 【ここを修正】層のインデックス i に応じて has_attn を切り替える
        # 例：1:1交互配置の場合 -> 偶数層(0, 2, 4...)は Attention あり(True)、奇数層(1, 3, 5...)はなし(False)
        self.layers = nn.ModuleList([
            DRNA_Block(
                d_model=d_model, 
                n_heads=n_heads, 
                head_dim=self.head_dim, 
                d_ff=d_ff, 
                has_attn=(i % 4 == 0)  # ここで 1:1 の交互配置を実現
                # 3層に1回だけAttentionにする(1/3配置)場合は has_attn=(i % 3 == 0) とできる
            ) 
            for i in range(n_layers)
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
            # どのブロック（Attentionあり / なし）であっても共通の回転場を流し込む
            x = layer(x, cos, sin, mask=mask)

        x = self.final_norm(x) # 出力前の最終同期
        return self.output_head(x)

'''
261018：O(N)^x 軽減対応のためATTNなし単独MLP層(SwiGLU型)を1/2以上交互挿入する
260528：SwiGLU構成／動的ゲート制御(d_ff で Dropout せず、w2 で d_model 凝縮直後で適用)
260520：maskの微調整(AMP対応)／MoE-LoRA版、vlayer版、D-RNAの活用例を汎用コード化
260507：Kによる回転で文脈に単語を沿わせ２重らせんの干渉による取捨選択とホログラム合成を可能にする
260505：model構成から学習解像度を自動化、汎用 mask の精度への適正化、RMSNormへの移行
260503：padding を引数で指定できるよう変更
    # 例：一般的な Tokenizer の pad_id が 0 の場合
    output = model(input_ids, pad_id=0)
    # 例：Hugging Face 等の tokenizer を使っている場合
    output = model(input_ids, pad_id=tokenizer.pad_token_id)
260502：変数名を正確化(head_dim)、汎用 mask に変更し padding 等に対応可
'''

'''
汎用型 D-RNA (Pre-Norm) License: Apache License 2.0 https://github.com/muooon/DRNA
Attention is all you need_started, Resonance is all you need_endure, 
Neocognitron ― Transformer ― D‑RNA Dream Resonance Never Adjourns — it goes on...
'''
