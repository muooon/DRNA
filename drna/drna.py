import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
D‑RNA: Dual‑Helix Resonance Neural Architecture (DRNA) Pre-Norm 版
260502：変数名を正確化(head_dim)、汎用 mask に変更し padding 等に対応可
仕様：GELU(Activation)、RoPE(head_dim)、mask(padding + causal)
Transformerの全接続性を継承しつつ、二重らせん(Dual-Helix)構造による
｢共鳴収縮｣(Resonant Contraction)を物理的に再現したニューラルアーキテクチャです
螺旋の同期：Attention(文脈の回想)とMLP(知識の定着)を直列に配置し RoPE で情報を同期
位相の保持：RoPE(Phase Field)を回転場として利用し、安定した相対位置を保ち早期収束を両立
高密度圧縮：Pre-Norm により、各らせんを安定的に収縮させ、全結合により記憶を定着させる
'''

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
    """位相回転の適用"""
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
        self.norm1 = nn.LayerNorm(d_model) # 演算の前に配置
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # らせんB: 記憶系 (MLP)
        self.norm2 = nn.LayerNorm(d_model) # 演算の前に配置
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
        
        # --- らせんA (Attention Resonance: Pre-Norm) ---
        residual = x
        x_norm = self.norm1(x) # 先にNormを計算
        
        qkv = self.qkv(x_norm).reshape(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q, k = apply_drna_rope(q, k, cos, sin)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            attn = attn + mask
        
        attn = F.softmax(attn, dim=-1)
        a_out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        
        # 残差接続（収縮ではなく蓄積）
        x = residual + self.dropout(self.out_proj(a_out))
        
        # --- らせんB (MLP Resonance: Pre-Norm) ---
        residual = x
        x_norm = self.norm2(x) # 先にNormを計算
        
        # 残差接続
        x = residual + self.dropout(self.mlp(x_norm))
        
        return x

class DRNA_Model(nn.Module):
    """汎用 DRNA モデルコンテナ（安定化 Pre-Norm 版）"""
    def __init__(self, vocab_size, d_model=256, n_layers=16, n_heads=8, d_ff=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head_dim = d_model // n_heads
        self.rope = DRNA_RoPE(self.head_dim)
        
        self.layers = nn.ModuleList([
            DRNA_Block(d_model, n_heads, self.head_dim, d_ff) for _ in range(n_layers)
        ])
        
        # Pre-Norm構造の場合、最終レイヤーの後に全体のNormを置くのが一般的
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        b, s = x.shape

        if mask is None:
            device = x.device
            pad_mask = (x != 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
            causal = torch.triu(torch.ones(s, s, device=device), diagonal=1).bool()
            causal = causal.unsqueeze(0).unsqueeze(0)      # (1,1,S,S)
            attn_mask = pad_mask & (~causal)               # (B,1,S,S)
            mask = attn_mask.masked_fill(~attn_mask, float('-inf'))

        cos, sin = self.rope(x, x.size(1))
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask)

        x = self.final_norm(x) # 出力前の最終同期
        return self.output_head(x)


'''
汎用型 D-RNA (Pre-Norm) License: Apache License 2.0 https://github.com/muooon/DRNA
Attention is all you need_started, Resonance is all you need_endure, 
Neocognitron ― Transformer ― D‑RNA Dream Resonance Never Adjourns — it goes on...
'''
