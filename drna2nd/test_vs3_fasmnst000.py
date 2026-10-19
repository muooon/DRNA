import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ==========================================
# 0. 共通シード設定関数
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache() # VRAMキャッシュもクリア

# ==========================================
# 共通部品 (RMSNorm & RoPE)
# ==========================================
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


# ==========================================
# 1. D-RNA モデル
# ==========================================
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
        self.pixel_embed = nn.Linear(1, d_model)
        self.head_dim = d_model // n_heads
        self.rope = DRNA_RoPE(self.head_dim)
        self.layers = nn.ModuleList([
            DRNA_Block(d_model, n_heads, self.head_dim, d_ff, has_attn=(i % 4 == 0))
            for i in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = self.pixel_embed(x)
        s = x.size(1)
        device = x.device
        causal = torch.triu(torch.ones(s, s, device=device), diagonal=1).bool().unsqueeze(0).unsqueeze(0)
        mask = torch.zeros((b, 1, s, s), device=device, dtype=x.dtype).masked_fill_(causal, -65500.0)

        cos, sin = self.rope(x, s)
        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask)
        return self.classifier(self.final_norm(x).mean(dim=1))


# ==========================================
# 2. 標準 Transformer モデル
# ==========================================
class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x

class StandardTransformerClassifier(nn.Module):
    def __init__(self, num_classes=10, d_model=128, n_layers=4, n_heads=4, d_ff=512, max_seq_len=784):
        super().__init__()
        self.pixel_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([StandardTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = self.pixel_embed(x) + self.pos_embed[:, :h*w, :]
        s = x.size(1)
        device = x.device
        mask = torch.triu(torch.full((s, s), float('-inf'), device=device), diagonal=1)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.classifier(self.final_norm(x).mean(dim=1))


# ==========================================
# 3. ハイブリッド Linear Attention モデル
# ==========================================
class HybridLinearAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, d_ff=None, dropout=0.1, has_attn=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.has_attn = has_attn

        if self.has_attn:
            self.norm1 = RMSNorm(d_model)
            self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            self.norm1 = RMSNorm(d_model)
            self.lin_qkv = nn.Linear(d_model, d_model * 3, bias=False)
            self.lin_out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        if d_ff is None:
            d_ff = int(2 * (d_model * 4) / 3)

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, s, d = x.shape
        residual = x

        if self.has_attn:
            x_norm1 = self.norm1(x)
            qkv = self.qkv(x_norm1).reshape(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if mask is not None:
                attn = attn + mask
            attn = F.softmax(attn, dim=-1)
            a_out = self.dropout(self.out_proj((attn @ v).transpose(1, 2).reshape(b, s, d)))
        else:
            x_norm1 = self.norm1(x)
            qkv = self.lin_qkv(x_norm1).reshape(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = F.elu(q) + 1.0
            k = F.elu(k) + 1.0

            context = torch.matmul(k.transpose(-2, -1), v)
            out = torch.matmul(q, context)
            normalizer = torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1))
            normalizer = torch.clamp(normalizer, min=1e-6)
            out = out / normalizer

            a_out = self.dropout(self.lin_out_proj(out.transpose(1, 2).reshape(b, s, d)))

        x_norm2 = self.norm2(x + a_out)
        gate = F.silu(self.w1(x_norm2))
        current_value = self.w3(x_norm2)
        m_out = self.dropout(self.w2(gate * current_value))

        return residual + a_out + m_out

class HybridLinearAttentionClassifier(nn.Module):
    def __init__(self, num_classes=10, d_model=128, n_layers=4, n_heads=4, d_ff=512):
        super().__init__()
        self.pixel_embed = nn.Linear(1, d_model)
        self.head_dim = d_model // n_heads
        self.layers = nn.ModuleList([
            HybridLinearAttentionBlock(d_model, n_heads, self.head_dim, d_ff, has_attn=(i % 4 == 0))
            for i in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = self.pixel_embed(x)
        s = x.size(1)
        device = x.device
        causal = torch.triu(torch.ones(s, s, device=device), diagonal=1).bool().unsqueeze(0).unsqueeze(0)
        mask = torch.zeros((b, 1, s, s), device=device, dtype=x.dtype).masked_fill_(causal, -65500.0)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        return self.classifier(x.mean(dim=1))


# ==========================================
# 4. 共通評価 & 実行メインスクリプト
# ==========================================
def evaluate_accuracy(model, testloader, device, max_samples=1000):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total >= max_samples:
                break
    model.train()
    return 100.0 * correct / total

def run_battle(model, trainloader, testloader, device, model_name="Model", total_steps=500, seed=42):
    # VRAMのピーク統計をリセットして正確に測定開始
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    print(f"\n==========================================")
    print(f"🥊 参戦: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} | Shared Seed: {seed}")
    print(f"==========================================")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    start_time = time.time()
    history = []
    data_iter = iter(trainloader)

    for step in range(1, total_steps + 1):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(trainloader)
            images, labels = next(data_iter)
            
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == total_steps:
            acc = evaluate_accuracy(model, testloader, device)
            history.append((step, loss.item(), acc))
            
            # 正確なVRAMピーク使用量（MB単位）を取得
            if device.type == "cuda":
                vram_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                vram_str = f" | VRAM Peak: {vram_used:.2f} MB"
            else:
                vram_str = ""
                
            print(f"[{model_name}] Step [{step}/{total_steps}] | Loss: {loss.item():.4f} | Test Acc: {acc:.2f}%{vram_str}")

    elapsed = time.time() - start_time
    print(f"🏁 {model_name} 完走！ (所要時間: {elapsed:.2f}秒)")
    return history

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 【変更点】固定の42にせず、実行ごとにランダムなシードを1つだけ生成して共有する
    COMMON_SEED = random.randint(0, 2**31 - 1)
    set_seed(COMMON_SEED)
    print(f"🎲 今回の実行用ランダム共通シードを [{COMMON_SEED}] に設定しました。")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    TOTAL_STEPS = 500

    # 1. D-RNA (回転場ありハイブリッド)
    drna_model = DRNA_ImageClassifier().to(device)
    drna_history = run_battle(drna_model, trainloader, testloader, device, model_name="D-RNA", total_steps=TOTAL_STEPS, seed=COMMON_SEED)

    # 2. Standard Transformer (フル関所)
    transformer_model = StandardTransformerClassifier().to(device)
    transformer_history = run_battle(transformer_model, trainloader, testloader, device, model_name="Standard Transformer", total_steps=TOTAL_STEPS, seed=COMMON_SEED)

    # 3. Hybrid Linear Attention (4層に1回フルAttn ＋ 残りLinear Attn)
    hybrid_linear_model = HybridLinearAttentionClassifier().to(device)
    hl_history = run_battle(hybrid_linear_model, trainloader, testloader, device, model_name="Hybrid Linear Attention", total_steps=TOTAL_STEPS, seed=COMMON_SEED)

    print("\n==========================================")
    print("🏆 真・3者ガチンコ対決 最終結果サマリー")
    print("==========================================")
    print(f"1. Standard Transformer        -> Acc: {transformer_history[-1][2]:.2f}% (Loss: {transformer_history[-1][1]:.4f})")
    print(f"2. D-RNA                       -> Acc: {drna_history[-1][2]:.2f}% (Loss: {drna_history[-1][1]:.4f})")
    print(f"3. Hybrid Linear Attention     -> Acc: {hl_history[-1][2]:.2f}% (Loss: {hl_history[-1][1]:.4f})")
    print(f"🎉 共通シード ({COMMON_SEED}) での公平な熱戦、お疲れ様でした！")