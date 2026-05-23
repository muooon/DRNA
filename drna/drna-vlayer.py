import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math

'''
D‑RNA: Dual‑Helix Resonance Neural Architecture (DRNA) Pre-Norm･Kv-RoPE･vlayer版
仕様：Pre-Norm(RMSNorm)、GELU(Activation)、Kv-RoPE(head_dim)、mask(padding + causal)
仮想レイヤ再帰（p_layers）、Gradient Checkpointing対応
'''

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
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
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        # らせんB: 記憶系 (MLP)
        self.norm2 = RMSNorm(d_model)
        d_ff = d_ff or d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        b, s, d = x.shape
        residual = x

        # らせんA (Attention) 並列方式
        x_norm1 = self.norm1(x)
        qkv = self.qkv(x_norm1).reshape(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # K因果的動的回転
        k_for_phase = torch.cat([torch.zeros_like(k[:, :, :1, :]), k[:, :, :-1, :]], dim=2)
        rt_phase = torch.tanh(k_for_phase) * math.pi

        # 静的RoPEを動的位相で変調
        d_cos = (cos * torch.cos(rt_phase)) - (sin * torch.sin(rt_phase))
        d_sin = (sin * torch.cos(rt_phase)) + (cos * torch.sin(rt_phase))

        # ２重らせん回転
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
        m_out = self.mlp(x_norm2)

        # 並列結合
        x = residual + self.dropout(a_out) + self.dropout(m_out)
        return x

class DRNA_Recurrent_Model(nn.Module):
    """汎用 DRNA モデルコンテナ（レイヤ回帰・仮想レイヤ組み込み版）"""
    def __init__(self, vocab_size, d_model=256, n_layers=4, p_layers=4, n_heads=8, d_ff=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head_dim = d_model // n_heads
        self.rope = DRNA_RoPE(self.head_dim)
        self.p_layers = p_layers  # 仮想レイヤーの周回数（再帰回数）

        self.layers = nn.ModuleList([
            DRNA_Block(d_model, n_heads, self.head_dim, d_ff) for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None, pad_id=None):
        b, s = x.shape
        device = x.device
        inputs = x
        x = self.embed(x)

        if mask is None or mask.sum() == 0:
            # 退避させた inputs でパディング位置を判定
            # パッドマスクとコーザルマスクを判定(pad_idの型チェック＆テンソルバグ修正)
            p_id = pad_id.item() if isinstance(pad_id, torch.Tensor) else pad_id
            pad_mask = (inputs != p_id).unsqueeze(1).unsqueeze(2) if isinstance(p_id, (int, float)) else torch.ones((1, 1, 1, s), device=device, dtype=torch.bool)
            causal = torch.triu(torch.ones(s, s, device=device), diagonal=1).bool().unsqueeze(0).unsqueeze(0)

            # 環境(fp16/32)に応じた最小値を安全に自動計算
            inf_value = torch.finfo(x.dtype).min if x.dtype != torch.float16 else -65500.0

            # ゼロ初期化テンソルに無効領域をインプレースで直接埋める(カンニングの完全遮断)
            mask = torch.zeros((b, 1, s, s), device=device, dtype=x.dtype).masked_fill_(causal | (~pad_mask), inf_value)

        cos, sin = self.rope(x, x.size(1))

        # 実レイヤーのループ
        for layer in self.layers:
            # 🌀 仮想レイヤーの再帰ループ
            for _ in range(self.p_layers):
                if self.training:
                    # 確実に勾配追跡を有効化するため、x のrequires_gradを確認・保障
                    if not x.requires_grad:
                        x.requires_grad_()
                    # グラディエント・チェックポインティングによるVRAM抑制
                    x = checkpoint.checkpoint(
                        layer,
                        x,
                        cos,
                        sin,
                        mask,
                        use_reentrant=False
                    )
                else:
                    x = layer(x, cos, sin, mask=mask)

        x = self.final_norm(x)
        return self.output_head(x)

'''
260520：p_layers による回帰で仮想レイヤをつくる(グラディエント・チェックポインティング活用/VRAM抑制)
｢仮想レイヤ｣は物理的なレイヤ数を最小限に、同一レイヤ内で自己周回(再帰)させ、安定的な効率化を実現する
グラディエント・チェックポインティング(GC)による再帰は中間計算(勾配)を必要時に再計算することで省VRAM化をします
'''

'''
汎用型 D-RNA (Pre-Norm) License: Apache License 2.0 https://github.com/muooon/DRNA
Attention is all you need_started, Resonance is all you need_endure, 
Neocognitron ― Transformer ― D‑RNA Dream Resonance Never Adjourns — it goes on...
'''