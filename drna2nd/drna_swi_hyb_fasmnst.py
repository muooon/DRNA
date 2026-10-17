import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math

# --- 1. 画像分類用 D-RNA モデル (drna_fminist.py 的なイメージ) ---

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(norm + self.eps))

class DRNA_RoPE(nn.Module):
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
    def __init__(self, d_model, n_heads, head_dim, d_ff=None, dropout=0.1, has_attn=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.has_attn = has_attn

        if self.has_attn:
            self.norm1 = RMSNorm(d_model)
            self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        if d_ff is None:
            d_ff = int(2 * (d_model * 4) / 3)
            
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        b, s, d = x.shape
        residual = x

        # らせんA (Attention)
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
            a_out = 0.0

        # らせんB (MLP)
        x_norm2 = self.norm2(x)
        gate = F.silu(self.w1(x_norm2))
        current_value = self.w3(x_norm2)
        m_out_raw = self.w2(gate * current_value)

        if not self.has_attn:
            m_reshaped = m_out_raw.reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
            def rotate_half(tensor):
                x1, x2 = tensor.chunk(2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)
            m_rotated = (m_reshaped * cos) + (rotate_half(m_reshaped) * sin)
            m_out_raw = m_rotated.transpose(1, 2).reshape(b, s, d)

        m_out = self.dropout(m_out_raw)
        return residual + a_out + m_out

class DRNA_ImageClassifier(nn.Module):
    def __init__(self, num_classes=10, d_model=128, n_layers=4, n_heads=4, d_ff=512):
        super().__init__()
        # 画像の1ピクセル(スカラー)を d_model 次元に射影
        self.pixel_embed = nn.Linear(1, d_model)
        self.head_dim = d_model // n_heads
        self.rope = DRNA_RoPE(self.head_dim)

        # 1:1 ハイブリッド層
        self.layers = nn.ModuleList([
            DRNA_Block(d_model, n_heads, self.head_dim, d_ff, has_attn=(i % 4 == 0))
            for i in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        # 画像全体をプールしてクラス分類するヘッド
        self.classifier = nn.Linear(d_model, num_classes)

        # 切断正規分布による初期化の適用
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: (B, 1, 28, 28) -> Flatten to (B, 784, 1)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c) # (B, 784, 1)
        
        x = self.pixel_embed(x) # (B, 784, d_model)
        s = x.size(1)
        
        # 因果マスクの作成 (画像用にも因果でピクセルを走査してみる)
        device = x.device
        causal = torch.triu(torch.ones(s, s, device=device), diagonal=1).bool().unsqueeze(0).unsqueeze(0)
        inf_value = -65500.0
        mask = torch.zeros((b, 1, s, s), device=device, dtype=x.dtype).masked_fill_(causal, inf_value)

        cos, sin = self.rope(x, s)
        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask)

        x = self.final_norm(x)
        # シーケンス方向の平均プールを取って画像全体の表現にする
        x_pooled = x.mean(dim=1)
        return self.classifier(x_pooled)


# --- 3. FashionMNIST 学習実行スクリプト ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データローダーの準備 (FashionMNIST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = DRNA_ImageClassifier().to(device)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("\n--- Starting FashionMNIST Training Epoch 1 ---")
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Batch [{i+1}/{len(trainloader)}] | Loss: {loss.item():.4f}")
        
        # テスト用に最初の200バッチだけ回して確認
        if i >= 300:
            break
    print("--- FashionMNIST Smoke Test Completed Successfully! ---")