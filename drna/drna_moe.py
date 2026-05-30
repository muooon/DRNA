import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
D‑RNA: Dual‑Helix Resonance Neural Architecture (DRNA) Pre-Norm･Kv-RoPE･MoE‑LoRA 版
仕様：Pre-Norm(RMSNorm)、GELU(Activation)、Kv-RoPE(head_dim)、mask(padding + causal)
汎用コードのコア(Base Weight)を不動の岩盤(ランダム初期化・完全固定)としつつ、
Attention および MLP の全線形層に LoRA による複数 expert を配置する D-RNA 応用型です
入力トークン、あるいはシーケンス特性に応じて動的に expert を選択・融合する MoE的拡張版
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

class MoELoRALinear(nn.Module):
    """固定されたベース線形層に対して、複数のLoRA Expertを動的にルーティングするMoE構造"""
    def __init__(self, in_features, out_features, r=16, lora_alpha=16, num_experts=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r
        self.num_experts = num_experts

        # 不動の岩盤 (ランダム初期化のまま勾配を固定)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.base_bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        nn.init.uniform_(self.base_bias, -bound, bound)

        # MoE-LoRA Experts (学習対象)
        self.lora_A = nn.Parameter(torch.randn(num_experts, r, in_features) / math.sqrt(in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, out_features, r))

        # どのExpertをどれだけ使うかを決定するルーター (シーケンス/トークン単位)
        self.router = nn.Linear(in_features, num_experts)

    def forward(self, x):
        # 共通のベース出力を計算
        base_out = F.linear(x, self.base_weight, self.base_bias)

        # ルーティング重みの計算 (Softmaxによるソフトな結合、またはTop-kへの拡張も可能)
        # x: (B, S, in_features) -> router_logits: (B, S, num_experts)
        router_logits = self.router(x)
        router_weights = F.softmax(router_logits, dim=-1).unsqueeze(-1) # (B, S, num_experts)

        # 全ExpertのLoRA出力をループなしで一斉に並列計算
        # self.lora_A の形状: (num_experts, r, in_features)
        lora_A_out = torch.einsum("bsi,ejr->bsej", x, self.lora_A)

        # self.lora_B の形状: (num_experts, out_features, r)
        # -> lora_out_all: (B, S, num_experts, out_features)
        lora_out_all = torch.einsum("bsej,ekj->bsek", lora_A_out, self.lora_B) * self.scaling
        # ルーターの重みで加重平均して合流
        # (B, S, num_experts, out_features) * (B, S, num_experts, 1) -> sum over experts
        lora_out_total = (lora_out_all * router_weights).sum(dim=2)

        return base_out + lora_out_total

class DRNA_MoE_Block(nn.Module):
    """DRNA共鳴ブロック：MoE-LoRA並列構造"""
    def __init__(self, d_model, n_heads, head_dim, d_ff=None, dropout=0.1, r=16, num_experts=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        # らせんA: 回想系 (Attention)
        self.norm1 = RMSNorm(d_model)
        self.qkv = MoELoRALinear(d_model, d_model * 3, r=r, num_experts=num_experts)
        self.out_proj = MoELoRALinear(d_model, d_model, r=r, num_experts=num_experts)

        # らせんB: 記憶系 (MLP)
        self.norm2 = RMSNorm(d_model)
        d_ff = d_ff or d_model * 4
        
        # 汎用コードの直列記述を維持するため、Sequentialではなく個別にMoEレイヤを定義
        self.mlp_in = MoELoRALinear(d_model, d_ff, r=r, num_experts=num_experts)
        self.mlp_out = MoELoRALinear(d_ff, d_model, r=r, num_experts=num_experts)
        self.activation = nn.GELU()
        self.mlp_dropout = nn.Dropout(dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        b, s, d = x.shape
        residual = x

        # らせんA (Attention)
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
        a_out = self.out_proj(a_out_raw)

        # らせんB (MLP)
        x_norm2 = self.norm2(x)
        m_intermediate = self.activation(self.mlp_in(x_norm2))
        m_out = self.mlp_out(self.mlp_dropout(m_intermediate))

        # 並列方式
        x = residual + self.dropout(a_out) + self.dropout(m_out)
        return x

class DRNA_MoE_Model(nn.Module):
    """汎用 DRNA モデルコンテナ（ランダム初期化ベース＋MoE-LoRA拡張・UTF-8対応版）"""
    def __init__(self, vocab_size=256, d_model=256, n_layers=16, n_heads=8, d_ff=1024, lora_r=16, num_experts=4):
        super().__init__()
        # UTF-8直受けのため、vocab_sizeは固定で256を指定可能
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head_dim = d_model // n_heads
        self.rope = DRNA_RoPE(self.head_dim)

        self.layers = nn.ModuleList([
            DRNA_MoE_Block(d_model, n_heads, self.head_dim, d_ff, r=lora_r, num_experts=num_experts) for _ in range(n_layers)
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

        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask)

        x = self.final_norm(x)
        return self.output_head(x)

'''
260520：LoRA による MoE(Mixture of Experts) 多重拡張化をする Plugin型です
学習元モデルをランダム初期化による完全カオス状態(凍結運用)とすることで"公開鍵"的に扱うことが可能となる
これにより LoRA は"秘密鍵"的に扱える、つまりこの学習元モデルでしか機能しないためセキュリティを堅持できる
学習元モデルを非公開で秘匿することで情報流出などを抑止した運用も可能になる
---
MoE-LoRAによる多重知性拡張－LoRA形式は高効率な学習と推論を可能とします
学習元モデル単体では具体的な知識や偏りを持たない安全な共通基盤になります(ランダム行列表なので)
このLoRAは学習元モデルでしか使えないため、LoRA単体だけではなにもできません
完全非公開・秘匿とすることで、情報流出リスクを極小化したクローズド運用も実現可能です
学習元モデルのみを公開するオープン運用でMoE-LoRAをみんなで自由に作成し機能向上を図ることも可能です
---
マルチモーダル対応：画像･動画･音声などの学習も可能です(必要ならトークナイザ等の差し替えも可能です)
VAEなどの外付けをせず、UTF8 を用いたトークンで 16x16 パッド化などでViT的な学習もできます
事前学習(プレトレーニング)もLoRAで行うことで高速化効率化を果たします(応答専用LoRAも学習可)
純粋な古代語LoRAなどをつくることで現代語に浸食されたり現代語を破壊するような干渉も防げます
'''

'''
[ 9B D-RNA ランダムベース重み (共通基盤：完全固定) ] (Dim4096/Layer32などを想定)
         ├──► 【言語LoRA (各国語/英語)】 ────► 日常対話・創造的執筆
         ├──► 【視覚パッチLoRA (512px)】 ────► 画像認識・マルチモーダル
         ├──► 【時間補間LoRA (fps拡張)】 ──► 滑らかな動画生成
         ├──► 【超解像LoRA (4K拡大)】 ────► 究極のディテール補間
         └──► 【聴覚パッチLoRA (音声波形)】 ─► 音声認識
画像：512px生成(主題と背景と物体等の対応関係)─►アップスケール(局所超解像MoE-LoRA)
動画：512px生成(8fps)─►フレーム補間(60fps)─►空間アップスケール(4K超解像)
このように位置関係LoRA、精細化LoRA、動体追従LoRA、などを付け足すだけで拡張可能です
言語も種類問わず、プログラム言語専用、国や地域の固有のもの、歴史的なもの、を干渉防止で学習可能です
'''

'''
汎用型 D-RNA (Pre-Norm) License: Apache License 2.0 https://github.com/muooon/DRNA
Attention is all you need_started, Resonance is all you need_endure, 
Neocognitron ― Transformer ― D‑RNA Dream Resonance Never Adjourns — it goes on...
'''
