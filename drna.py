import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
D‑RNA: Dual‑Helix Resonance Neural Architecture (DRNA) 260420
Transformerの全接続性を継承しつつ、二重らせん(Dual-Helix)構造による
｢共鳴収縮｣(Resonant Contraction)を物理的に再現したニューラルアーキテクチャです
螺旋の同期: Attention(文脈の回想)とMLP(知識の定着)を直列に配置し、情報を一段ずつ絞り込む
位相の保持: RoPE(Rotary Positional Embedding)を回転場として利用し、相対位置を保つ
高密度圧縮: Post-Norm構造により、各らせんの出力直後に情報を収縮させ、意味を確定させる
'''

class DRNA_RoPE(nn.Module):
    """二重らせんの位相を決定する回転場"""
    def __init__(self, d_model, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]

def apply_drna_rope(q, k, cos, sin):
    """位相回転の適用"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    cos, sin = cos[:, None, :, :], sin[:, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class DRNA_Block(nn.Module):
    """DRNA共鳴ブロック：元設計に忠実な直列共鳴構造"""
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # らせんA: 回想系 (Attention)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # らせんB: 記憶系 (MLP)
        d_ff = d_ff or d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        b, s, d = x.shape
        
        # --- らせんA (Attention Resonance) ---
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q, k = apply_drna_rope(q, k, cos, sin)
        
        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        attn = F.softmax(attn, dim=-1)
        
        a_out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        
        # 収縮統合1
        x = self.norm1(x + self.dropout(self.out_proj(a_out)))
        
        # --- らせんB (MLP Resonance) ---
        # 収縮統合2
        x = self.norm2(x + self.dropout(self.mlp(x)))
        
        return x

class DRNA_Model(nn.Module):
    """汎用 DRNA モデルコンテナ"""
    def __init__(self, vocab_size, d_model=256, n_layers=16, n_heads=8, d_ff=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = DRNA_RoPE(d_model // n_heads)
        
        self.layers = nn.ModuleList([
            DRNA_Block(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        cos, sin = self.rope(x, x.size(1))
        x = self.embed(x)
        
        for layer in self.layers:
            x = layer(x, cos, sin)
            
        return self.output_head(x)

'''
汎用型 D-RNA コード License: Apache License 2.0
https://github.com/muooon/DRNA
'''
