# D-RNA：Dual‑Helix Resonance Neural Architecture (DRNA)  
### Pre-Norm Edition  

### Attention is all you need_started,  
### Resonance is all you need_endure,  
### Neocognitron ― Transformer ― Dream Resonance Never Adjourns — it goes on...  

⭐ If you like this project, please give it a star ⭐  
readme：[English](README.md) | [日本語](README_JA.md)  

D-RNA is a new neural architecture centered on a dual helix structure and a rotation field produced by RoPE.  

In this architecture, Attention and MLP are synchronized into a dual helix, and information is holographically compressed through Resonant Contraction.  
This method rearranges sparse representations into dense ones to achieve high expressiveness using the depth‑direction structure alone, without increasing the number of dimensions.  
A key feature of this approach is its ability to preserve the full connectivity of the Transformer architecture while suppressing catastrophic forgetting and retaining subtle fluctuations and phase information.  

---

### Explanation  
High‑Density Transformer and Fast Convergence via Dual‑Helix Resonant Contraction Architecture (D‑RNA)  

#### [D-RNA-paper(article)](https://huggingface.co/muooon/DRNA/raw/main/D%E2%80%91RNA_paper_260507(ENG).txt) 

---

### Features  
High structural compatibility: It has the exact same input–output shape as a standard Transformer Block, allowing it to be smoothly substituted as the core of an architecture.  
Resonant Contraction: By synchronizing Attention and the MLP in a double‑helix pattern and converging information into a phase field, it dramatically increases representational density.  
Depth as an alternative to dimensionality: The spiral rotation (depth‑wise operations) compensates for limited dimensionality and enables holographic information retention without increasing parameter count.  
Excellent learning efficiency: The spiral‑based information attraction (synchronization) achieves astonishing early convergence with far fewer steps than a Transformer.  
Fine‑grained phase preservation: The rotational field powered by RoPE preserves subtle fluctuations and relative contextual relationships that are often lost in conventional architectures.  
Re‑synchronization of knowledge: Existing weights can be transplanted as initialization and gently adapted to the spiral phase with a low learning rate, allowing existing intelligence to be evolved or overwritten into the D-RNA structure.


### Notes  
Optimization of learning rate (LR):  
Because D-RNA synchronizes information extremely quickly through Resonant Contraction, it converges sufficiently — and rapidly — even with a lower learning rate compared to a standard Transformer.  
If the LR is set too high, the resonance may be excessively amplified and cause oscillation, so starting with a modest LR is recommended.  
Synergistic gradient effects:  
Since Attention (recall) and the MLP (memory) are synchronized in a double‑helix sequence, Each weight update exerts a significant impact on synchronization.  
This is an advantage for fast convergence, but it also means that careful updates are key to stability.  
Parameter commonality:  
Hyperparameters such as weight initialization seeds and batch size can be inherited directly from standard Transformer settings.  

<details>

<summary> Characteristics of D-RNA </summary>

D-RNA constructs a resonant contraction method (resonant projection field) based on the “phase of the helix.”  
By transforming sparse structures into dense ones, this approach suppresses destructive forgetting (without causing mutual interference) and accelerates toward the shortest path.  
Even fine noise facilitates information purification, smoothing the manifold and cumulatively achieving generalization.  
These mechanisms are independent of any specific framework and function across all optimizers and models.  
In a sense, it is a mechanism resembling a biological brain, consisting of neuron- and glia-like structures.  
The resonant contraction method (resonant projection field) ultimately yields an equivalent of an ODE reduction approximation.  

</details>

---

### Conceptual Diagram

```
	Synchronizing “searching” (Attention)  
	   and “knowing” (MLP) in the phase of a spiral.  

	RoPE Rotation Field (Phase-Preserving)  
	Holographic Compression: Turning Sparse into Dense  

		A     M  
		 \   /  
		  \ /    ← This is Resonance  
		  / \      Synchronization occurs naturally through the seed  
		 /   \     Naturally, meaning emerges through a chain of synchronicities  
		A     M  

	Repeats in the depth direction to form a dual helix  
	(acts as a substitute for increasing dimensionality)  
```
---

### Minimal Block  

```python
class ResonantBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.mlp = MLP(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.n_heads = n_heads
        self.d_head = dim // n_heads

    def forward(self, x, cos, sin):
        # --- Attention Path (Pre-Norm) ---
        residual = x
        x_norm = self.norm1(x)  # 演算の前にNormを適用
        
        q, k, v = project_qkv(x_norm, self.qkv, self.n_heads, self.d_head)
        q, k = apply_rope(q, k, cos, sin)
        attn_out = attention(q, k, v)
        x = residual + self.out(attn_out)

        # --- MLP Path (Pre-Norm) ---
        residual = x
        x = residual + self.mlp(self.norm2(x))  # 演算の前にNormを適用
        
        return x
```

---

<details>

<summary> Replacement and Utilization of D-RNA </summary>

### Example: Replacing a Transformer block with a D-RNA block  

```python
class DRNA_ResonantBlock(nn.Module):
    """
    Replace the existing TransformerBlock with this ResonantBlock.
    I/O: [Batch, Seq, Dim] -> [Batch, Seq, Dim] (Fully compatible)
    Architecture: Pre-Norm (Stability-first for Deep Networks)
    """
    def __init__(self, dim, n_heads, mlp_dim_forward=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        
        # 1. Spiral Projection Layer (A)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        
        # 2. Spiral Memory Layer (B)
        mlp_dim = mlp_dim_forward if mlp_dim_forward else dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
        # 3. Normalization layer for pre-processing
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, cos, sin):
        """
        Phase information for RoPE as an argument (cos, sin)
        """
        # --- Attention Path (Pre-Norm) ---
        # Normalize -> QKV -> RoPE -> Residual Add
        residual = x
        x_norm = self.norm1(x)
        
        q, k, v = project_qkv(x_norm, self.qkv, self.n_heads, self.d_head)
        q, k = apply_rope(q, k, cos, sin)
        
        attn_out = attention(q, k, v)
        x = residual + self.out(attn_out) 

        # --- MLP Path (Pre-Norm) ---
        # Normalize -> MLP -> Residual Add
        residual = x
        x = residual + self.mlp(self.norm2(x)) 
        
        return x
```

### Replacement and Utilization of D-RNA  
A direct drop‑in replacement is not possible, but it can be utilized through “redefinition and re‑synchronization.”  
Why it cannot be used as‑is:  
While a standard Transformer stores information using an “absolute address” (absolute position), D-RNA processes information using the “phase of a spiral” (relative position), meaning the coordinate systems are fundamentally different.  
Even if the weights are copied directly, the phases do not align and no resonance cannot be induced immediately.  
How to replace it (implementation):  
The network’s input–output shapes are fully compatible.  
By rewriting the existing layers as ResonantBlock and migrating positional information into RoPE’s rotational field, the core upgrade is complete.  
How to utilize and adapt it (training):  
After transferring the existing model’s weights as initialization, continue training with a low learning rate.  
The previously static knowledge (existing weights) begins to synchronize with the spiral rotation, gradually blending into D-RNA’s “Resonant Contraction” process and evolving beyond the original performance.  

</details>

---

## BPC Comparison Chart  

Latest Test Results (Suitable for Learning)  
<img width="800" height="500" alt="bpc_prenorm_battle_Lay16-4x07x5000" src="https://github.com/user-attachments/assets/7239f630-e890-4997-93ff-7495e5c7af81" />  

AdamW：5e-5  
Vanilla:16L, VRAM:2.11GB, Step 5000 | BPC: 3.2970 | 801.9s  
D-RNA:4L, VRAM:1.03GB, Step 5000 | BPC: 2.8744 | 316.4s  
|Efficiency| VRAM: Reduced by approximately 50%, BPC: Improved accuracy, Speed: Approximately 2.5 times faster  

<details>

<summary> Log after Kv-RoPE restriction </summary>

Learning Test Status (Details):  
Model scale：dim：128、 layers：16/4(D-RNA)、 heads：4  
Data set：enwik8(100MB)  
Learning Settings：step：5,000、 batch：16、 seq_len：512、 AdamW(LR：5e-5)  

```prompt
Configured: Battle Mode (Vanilla=16L vs D-RNA=4L)  

--- 🚀 Starting Run: Transformer (Layers: 16) ---  
Step    0 | BPC: 8.3963 | VRAM: 2.09GB | 0.3s
Step   50 | BPC: 5.9260 | VRAM: 2.11GB | 8.3s
Step  100 | BPC: 5.4452 | VRAM: 2.11GB | 16.3s
Step  150 | BPC: 5.0283 | VRAM: 2.11GB | 24.1s
Step  200 | BPC: 4.5926 | VRAM: 2.11GB | 32.2s
Step  250 | BPC: 4.4936 | VRAM: 2.11GB | 40.2s
Step  300 | BPC: 4.4320 | VRAM: 2.11GB | 48.0s
Step  350 | BPC: 4.4243 | VRAM: 2.11GB | 55.9s
Step  400 | BPC: 4.0412 | VRAM: 2.11GB | 63.9s
Step  450 | BPC: 4.0397 | VRAM: 2.11GB | 71.8s
Step  500 | BPC: 4.1921 | VRAM: 2.11GB | 79.7s
Step  550 | BPC: 4.0418 | VRAM: 2.11GB | 87.6s
Step  600 | BPC: 4.1054 | VRAM: 2.11GB | 95.5s
Step  650 | BPC: 3.8973 | VRAM: 2.11GB | 103.5s
Step  700 | BPC: 4.0361 | VRAM: 2.11GB | 111.4s
Step  750 | BPC: 3.8873 | VRAM: 2.11GB | 119.4s
Step  800 | BPC: 3.8468 | VRAM: 2.11GB | 127.6s
Step  850 | BPC: 3.9349 | VRAM: 2.11GB | 135.8s
Step  900 | BPC: 3.9785 | VRAM: 2.11GB | 144.0s
Step  950 | BPC: 3.8893 | VRAM: 2.11GB | 152.2s
Step 1000 | BPC: 3.7580 | VRAM: 2.11GB | 160.4s
Step 1050 | BPC: 3.9328 | VRAM: 2.11GB | 168.5s
Step 1100 | BPC: 3.7746 | VRAM: 2.11GB | 176.6s
Step 1150 | BPC: 3.7990 | VRAM: 2.11GB | 184.7s
Step 1200 | BPC: 3.7760 | VRAM: 2.11GB | 193.0s
Step 1250 | BPC: 3.8704 | VRAM: 2.11GB | 201.2s
Step 1300 | BPC: 3.7458 | VRAM: 2.11GB | 209.4s
Step 1350 | BPC: 3.7624 | VRAM: 2.11GB | 217.6s
Step 1400 | BPC: 3.7851 | VRAM: 2.11GB | 225.6s
Step 1450 | BPC: 3.7754 | VRAM: 2.11GB | 233.6s
Step 1500 | BPC: 3.7048 | VRAM: 2.11GB | 241.6s
Step 1550 | BPC: 3.8543 | VRAM: 2.11GB | 249.6s
Step 1600 | BPC: 3.7900 | VRAM: 2.11GB | 257.6s
Step 1650 | BPC: 3.7374 | VRAM: 2.11GB | 265.6s
Step 1700 | BPC: 3.5948 | VRAM: 2.11GB | 273.6s
Step 1750 | BPC: 3.5474 | VRAM: 2.11GB | 281.6s
Step 1800 | BPC: 3.5863 | VRAM: 2.11GB | 289.7s
Step 1850 | BPC: 3.7306 | VRAM: 2.11GB | 297.7s
Step 1900 | BPC: 3.6679 | VRAM: 2.11GB | 305.7s
Step 1950 | BPC: 3.6901 | VRAM: 2.11GB | 313.7s
Step 2000 | BPC: 3.6446 | VRAM: 2.11GB | 321.7s
Step 2050 | BPC: 3.5935 | VRAM: 2.11GB | 329.7s
Step 2100 | BPC: 3.5685 | VRAM: 2.11GB | 337.7s
Step 2150 | BPC: 3.7369 | VRAM: 2.11GB | 345.7s
Step 2200 | BPC: 3.6565 | VRAM: 2.11GB | 353.7s
Step 2250 | BPC: 3.7226 | VRAM: 2.11GB | 361.7s
Step 2300 | BPC: 3.4056 | VRAM: 2.11GB | 369.7s
Step 2350 | BPC: 3.6761 | VRAM: 2.11GB | 377.6s
Step 2400 | BPC: 3.5442 | VRAM: 2.11GB | 385.6s
Step 2450 | BPC: 3.6574 | VRAM: 2.11GB | 393.6s
Step 2500 | BPC: 3.4996 | VRAM: 2.11GB | 401.6s
Step 2550 | BPC: 3.5436 | VRAM: 2.11GB | 409.6s
Step 2600 | BPC: 3.6407 | VRAM: 2.11GB | 417.6s
Step 2650 | BPC: 3.5530 | VRAM: 2.11GB | 425.6s
Step 2700 | BPC: 3.5134 | VRAM: 2.11GB | 433.6s
Step 2750 | BPC: 3.6320 | VRAM: 2.11GB | 441.6s
Step 2800 | BPC: 3.5229 | VRAM: 2.11GB | 449.6s
Step 2850 | BPC: 3.6339 | VRAM: 2.11GB | 457.6s
Step 2900 | BPC: 3.5928 | VRAM: 2.11GB | 465.6s
Step 2950 | BPC: 3.6163 | VRAM: 2.11GB | 473.6s
Step 3000 | BPC: 3.3798 | VRAM: 2.11GB | 481.6s
Step 3050 | BPC: 3.5823 | VRAM: 2.11GB | 489.6s
Step 3100 | BPC: 3.5384 | VRAM: 2.11GB | 497.6s
Step 3150 | BPC: 3.4950 | VRAM: 2.11GB | 505.6s
Step 3200 | BPC: 3.5007 | VRAM: 2.11GB | 513.6s
Step 3250 | BPC: 3.4352 | VRAM: 2.11GB | 521.6s
Step 3300 | BPC: 3.5145 | VRAM: 2.11GB | 529.6s
Step 3350 | BPC: 3.5518 | VRAM: 2.11GB | 537.7s
Step 3400 | BPC: 3.5272 | VRAM: 2.11GB | 545.7s
Step 3450 | BPC: 3.5821 | VRAM: 2.11GB | 553.7s
Step 3500 | BPC: 3.5452 | VRAM: 2.11GB | 561.7s
Step 3550 | BPC: 3.4426 | VRAM: 2.11GB | 569.7s
Step 3600 | BPC: 3.5087 | VRAM: 2.11GB | 577.7s
Step 3650 | BPC: 3.4893 | VRAM: 2.11GB | 585.7s
Step 3700 | BPC: 3.6078 | VRAM: 2.11GB | 593.7s
Step 3750 | BPC: 3.6168 | VRAM: 2.11GB | 601.7s
Step 3800 | BPC: 3.3611 | VRAM: 2.11GB | 609.7s
Step 3850 | BPC: 3.5110 | VRAM: 2.11GB | 617.7s
Step 3900 | BPC: 3.4627 | VRAM: 2.11GB | 625.7s
Step 3950 | BPC: 3.2842 | VRAM: 2.11GB | 633.7s
Step 4000 | BPC: 3.5764 | VRAM: 2.11GB | 641.7s
Step 4050 | BPC: 3.2557 | VRAM: 2.11GB | 649.7s
Step 4100 | BPC: 3.4295 | VRAM: 2.11GB | 657.7s
Step 4150 | BPC: 3.4520 | VRAM: 2.11GB | 665.7s
Step 4200 | BPC: 3.2938 | VRAM: 2.11GB | 673.8s
Step 4250 | BPC: 3.3882 | VRAM: 2.11GB | 681.8s
Step 4300 | BPC: 3.3491 | VRAM: 2.11GB | 689.8s
Step 4350 | BPC: 3.4648 | VRAM: 2.11GB | 697.8s
Step 4400 | BPC: 3.4442 | VRAM: 2.11GB | 705.8s
Step 4450 | BPC: 3.3809 | VRAM: 2.11GB | 713.8s
Step 4500 | BPC: 3.5511 | VRAM: 2.11GB | 721.8s
Step 4550 | BPC: 3.3884 | VRAM: 2.11GB | 729.8s
Step 4600 | BPC: 3.3117 | VRAM: 2.11GB | 737.8s
Step 4650 | BPC: 3.3749 | VRAM: 2.11GB | 745.8s
Step 4700 | BPC: 3.3855 | VRAM: 2.11GB | 753.8s
Step 4750 | BPC: 3.4674 | VRAM: 2.11GB | 761.9s
Step 4800 | BPC: 3.4271 | VRAM: 2.11GB | 769.9s
Step 4850 | BPC: 3.4085 | VRAM: 2.11GB | 777.9s
Step 4900 | BPC: 3.4258 | VRAM: 2.11GB | 785.9s
Step 4950 | BPC: 3.3319 | VRAM: 2.11GB | 793.9s
Step 5000 | BPC: 3.2970 | VRAM: 2.11GB | 801.9s

--- 🚀 Starting Run: D-RNA_Transformer(L4) (Layers: 4) ---  
Step    0 | BPC: 8.1803 | VRAM: 1.02GB | 0.1s
Step   50 | BPC: 6.0895 | VRAM: 1.03GB | 3.2s
Step  100 | BPC: 5.5667 | VRAM: 1.03GB | 6.3s
Step  150 | BPC: 5.0683 | VRAM: 1.03GB | 9.5s
Step  200 | BPC: 4.8225 | VRAM: 1.03GB | 12.6s
Step  250 | BPC: 4.5308 | VRAM: 1.03GB | 15.7s
Step  300 | BPC: 4.3597 | VRAM: 1.03GB | 18.8s
Step  350 | BPC: 4.3135 | VRAM: 1.03GB | 21.9s
Step  400 | BPC: 4.0541 | VRAM: 1.03GB | 25.0s
Step  450 | BPC: 4.0538 | VRAM: 1.03GB | 28.1s
Step  500 | BPC: 3.8305 | VRAM: 1.03GB | 31.3s
Step  550 | BPC: 3.9215 | VRAM: 1.03GB | 34.4s
Step  600 | BPC: 4.0164 | VRAM: 1.03GB | 37.5s
Step  650 | BPC: 3.8336 | VRAM: 1.03GB | 40.6s
Step  700 | BPC: 3.7699 | VRAM: 1.03GB | 43.7s
Step  750 | BPC: 3.8394 | VRAM: 1.03GB | 46.8s
Step  800 | BPC: 3.8393 | VRAM: 1.03GB | 49.9s
Step  850 | BPC: 3.7473 | VRAM: 1.03GB | 53.1s
Step  900 | BPC: 3.5263 | VRAM: 1.03GB | 56.2s
Step  950 | BPC: 3.6108 | VRAM: 1.03GB | 59.3s
Step 1000 | BPC: 3.6208 | VRAM: 1.03GB | 62.4s
Step 1050 | BPC: 3.4813 | VRAM: 1.03GB | 65.5s
Step 1100 | BPC: 3.6377 | VRAM: 1.03GB | 68.6s
Step 1150 | BPC: 3.5227 | VRAM: 1.03GB | 71.8s
Step 1200 | BPC: 3.5667 | VRAM: 1.03GB | 74.9s
Step 1250 | BPC: 3.4331 | VRAM: 1.03GB | 78.0s
Step 1300 | BPC: 3.4172 | VRAM: 1.03GB | 81.2s
Step 1350 | BPC: 3.6982 | VRAM: 1.03GB | 84.3s
Step 1400 | BPC: 3.3116 | VRAM: 1.03GB | 87.5s
Step 1450 | BPC: 3.4180 | VRAM: 1.03GB | 90.6s
Step 1500 | BPC: 3.5096 | VRAM: 1.03GB | 93.7s
Step 1550 | BPC: 3.3789 | VRAM: 1.03GB | 96.9s
Step 1600 | BPC: 3.3193 | VRAM: 1.03GB | 100.0s
Step 1650 | BPC: 3.2843 | VRAM: 1.03GB | 103.2s
Step 1700 | BPC: 3.3066 | VRAM: 1.03GB | 106.3s
Step 1750 | BPC: 3.2612 | VRAM: 1.03GB | 109.4s
Step 1800 | BPC: 3.2183 | VRAM: 1.03GB | 112.6s
Step 1850 | BPC: 3.2831 | VRAM: 1.03GB | 115.9s
Step 1900 | BPC: 3.3514 | VRAM: 1.03GB | 119.1s
Step 1950 | BPC: 3.3732 | VRAM: 1.03GB | 122.3s
Step 2000 | BPC: 3.3886 | VRAM: 1.03GB | 125.5s
Step 2050 | BPC: 3.3236 | VRAM: 1.03GB | 128.8s
Step 2100 | BPC: 3.4354 | VRAM: 1.03GB | 132.1s
Step 2150 | BPC: 3.0614 | VRAM: 1.03GB | 135.3s
Step 2200 | BPC: 3.2231 | VRAM: 1.03GB | 138.5s
Step 2250 | BPC: 3.1392 | VRAM: 1.03GB | 141.6s
Step 2300 | BPC: 3.2459 | VRAM: 1.03GB | 144.7s
Step 2350 | BPC: 3.0381 | VRAM: 1.03GB | 147.9s
Step 2400 | BPC: 3.2124 | VRAM: 1.03GB | 151.0s
Step 2450 | BPC: 3.0759 | VRAM: 1.03GB | 154.2s
Step 2500 | BPC: 3.1911 | VRAM: 1.03GB | 157.4s
Step 2550 | BPC: 3.2409 | VRAM: 1.03GB | 160.6s
Step 2600 | BPC: 3.1085 | VRAM: 1.03GB | 163.8s
Step 2650 | BPC: 3.2135 | VRAM: 1.03GB | 166.9s
Step 2700 | BPC: 3.1824 | VRAM: 1.03GB | 170.1s
Step 2750 | BPC: 3.0541 | VRAM: 1.03GB | 173.2s
Step 2800 | BPC: 3.2042 | VRAM: 1.03GB | 176.4s
Step 2850 | BPC: 3.2427 | VRAM: 1.03GB | 179.6s
Step 2900 | BPC: 3.1356 | VRAM: 1.03GB | 182.8s
Step 2950 | BPC: 3.1764 | VRAM: 1.03GB | 185.9s
Step 3000 | BPC: 3.2040 | VRAM: 1.03GB | 189.0s
Step 3050 | BPC: 3.1078 | VRAM: 1.03GB | 192.2s
Step 3100 | BPC: 3.0288 | VRAM: 1.03GB | 195.4s
Step 3150 | BPC: 3.0628 | VRAM: 1.03GB | 198.5s
Step 3200 | BPC: 3.2522 | VRAM: 1.03GB | 201.7s
Step 3250 | BPC: 3.0266 | VRAM: 1.03GB | 204.9s
Step 3300 | BPC: 3.0467 | VRAM: 1.03GB | 208.0s
Step 3350 | BPC: 3.0561 | VRAM: 1.03GB | 211.2s
Step 3400 | BPC: 3.0182 | VRAM: 1.03GB | 214.4s
Step 3450 | BPC: 3.0035 | VRAM: 1.03GB | 217.5s
Step 3500 | BPC: 3.0790 | VRAM: 1.03GB | 220.7s
Step 3550 | BPC: 3.0263 | VRAM: 1.03GB | 223.8s
Step 3600 | BPC: 3.0813 | VRAM: 1.03GB | 226.9s
Step 3650 | BPC: 3.1324 | VRAM: 1.03GB | 230.1s
Step 3700 | BPC: 3.1179 | VRAM: 1.03GB | 233.2s
Step 3750 | BPC: 3.1641 | VRAM: 1.03GB | 236.4s
Step 3800 | BPC: 3.0669 | VRAM: 1.03GB | 239.5s
Step 3850 | BPC: 3.1459 | VRAM: 1.03GB | 242.7s
Step 3900 | BPC: 2.8818 | VRAM: 1.03GB | 245.8s
Step 3950 | BPC: 2.9704 | VRAM: 1.03GB | 249.1s
Step 4000 | BPC: 3.0188 | VRAM: 1.03GB | 252.4s
Step 4050 | BPC: 2.9833 | VRAM: 1.03GB | 255.7s
Step 4100 | BPC: 3.2226 | VRAM: 1.03GB | 259.0s
Step 4150 | BPC: 3.1744 | VRAM: 1.03GB | 262.4s
Step 4200 | BPC: 2.9893 | VRAM: 1.03GB | 265.7s
Step 4250 | BPC: 3.1178 | VRAM: 1.03GB | 269.0s
Step 4300 | BPC: 2.9596 | VRAM: 1.03GB | 272.2s
Step 4350 | BPC: 3.1703 | VRAM: 1.03GB | 275.4s
Step 4400 | BPC: 2.8626 | VRAM: 1.03GB | 278.5s
Step 4450 | BPC: 2.9154 | VRAM: 1.03GB | 281.7s
Step 4500 | BPC: 2.9000 | VRAM: 1.03GB | 284.8s
Step 4550 | BPC: 3.0336 | VRAM: 1.03GB | 288.0s
Step 4600 | BPC: 3.0229 | VRAM: 1.03GB | 291.2s
Step 4650 | BPC: 3.1241 | VRAM: 1.03GB | 294.4s
Step 4700 | BPC: 3.0505 | VRAM: 1.03GB | 297.5s
Step 4750 | BPC: 3.1495 | VRAM: 1.03GB | 300.7s
Step 4800 | BPC: 3.0456 | VRAM: 1.03GB | 303.8s
Step 4850 | BPC: 2.9345 | VRAM: 1.03GB | 307.0s
Step 4900 | BPC: 3.1072 | VRAM: 1.03GB | 310.1s
Step 4950 | BPC: 2.9741 | VRAM: 1.03GB | 313.2s
Step 5000 | BPC: 2.8744 | VRAM: 1.03GB | 316.4s
```

</details>


<details>

<summary> Preview Kv-RoPE Restriction Test Results </summary>

||| Phase 1: Pure Structural Comparison (Same Number of Layers: 16L vs. 16L) |||

use-mask  
<img width="800" height="500" alt="bpc_prenorm_battle" src="https://github.com/user-attachments/assets/d599d19b-48b7-45e7-aff4-195067823976" />

use-mask 5000step
<img width="800" height="500" alt="bpc_prenorm_battle_5000" src="https://github.com/user-attachments/assets/7941f635-b0a7-471d-819f-76c9d55a5bc7" />
 

Learning Test Status (Details):  
Model Scale: Dimension (d_model): 256, Layers (n_layers): 16, Heads (n_heads): 8  
Dataset: enwik8 (100MB)  
Training Settings: Steps: 5,000, Batch Size: 16, Sequence Length: 512, AdamW (LR: 1e-4)  

Training Result Analysis (Overview):  
Training Efficiency: 30% improvement (Step efficiency: approx. 1.5x)  
Convergence Speed: 30% reduction in time cost (Convergence rate accelerated by 1.5x)  

Observed Benefits from Testing (Summary):  
Optimization of Parameter Density  
Structural advantage in convergence characteristics  
Expansion of Information Capacity within the same computational budget  
※ D-RNA evolves relative to the Transformer by utilizing phase synchronization via its helical structure, incurring only minimal internal computational overhead.  

| Metric                 | Normal Transformer | D-RNA Transformer | Difference / Efficiency      |  
|------------------------|--------------------|-------------------|------------------------------|  
| Steps to Reach Target  | 3,850 steps        | 2,350 steps       | ~39.0% reduction             |  
| Time Required          | 1365.4 sec         | 876.1 sec         | ~35.8% faster                |  
| VRAM Usage             | 4.51 GB            | 5.05 GB           | +0.54 GB cost   |  


||| Phase 2: Implementation of Optimization (Reducing the Number of Layers by Half: 16L vs. 8L) |||

use-mask 5000step  
<img width="800" height="500" alt="bpc_prenorm_battle_5000xcos" src="https://github.com/user-attachments/assets/8a75106e-65c5-4e2b-b842-39d0b92a1a00" />

use-mask 10000step  
<img width="800" height="500" alt="bpc_prenorm_battle_10000xcosGeLU" src="https://github.com/user-attachments/assets/0e3d2f9e-5507-4ec2-9349-deb4d47cfbe5" />  

Learning Test Status (Details):  
Model Scale: Dimension (d_model): 256, Layers (n_layers): 16 / 8(D-RNA), Heads (n_heads): 8  
Dataset: enwik8 (100MB)  
Training Settings: Steps: 10,000, Batch Size: 16, Sequence Length: 512, AdamW (LR: 1e-4), CosineAnnealing   

Training Result Analysis (Overview):  
Training Efficiency: 30% improvement (Step efficiency: approx. 1.5x)  
※ A reduction of up to approximately 60% in the time required to reach the same level of Perplexity (around BPC 2.05) was observed.  
Convergence Speed: 30% reduction in time cost (Convergence rate accelerated by 1.5x)  
※ By reducing the number of physical layers (from 16L to 8L) and employing a spiral structure (phase synchronization), we have achieved both optimization of parameter density and increased processing speed.  

Observed Benefits from Testing (Summary):  
Optimization of Parameter Density  
Structural advantage in convergence characteristics  
Expansion of Information Capacity within the same computational budget  
※ D-RNA evolves relative to the Transformer by utilizing phase synchronization via its helical structure, incurring only minimal internal computational overhead.  

| Metric                 | Normal Transformer | D-RNA Transformer | Difference / Efficiency      |  
|------------------------|--------------------|-------------------|------------------------------|  
| Steps to Reach Target  | 4,650 steps        | 3,650 steps       | ~21.0% reduction       |  
| Time Required          | 1641.5 sec         | 685.2 sec         | ~58.0% faster          |  
| VRAM Usage             | 4.79 GB            | 2.56 GB           | ~46.0% reduction  |  
| Final BPC       | 1.9272          | 1.8958          | Higher accuracy per layer     |  
| Final step time     | 3,542.2 sec         | 1,876.8 sec        | ~53.0% reduction      |    

Basic Architecture Specifications:  
Transformer   Absolute(Learned)   GELU  
D-RNA   RoPE(Rotary)   GELU  

</details>

---

New Perspective: Digital Vector × Phase Distance  

D-RNA can achieve high-resolution approximation even in low-bit environments (e.g., 1.58-bit / Ternary weights).  

Discrete Vectors: Each layer handles discrete, "digital" 3-value vectors (-1, 0, 1).  

<img width="800" height="400" alt="smooth" src="https://github.com/user-attachments/assets/69efc1cd-e13b-400f-a188-b886eb642f26" />

Continuous Distance: By stacking layers with different phases (Double Helix), these discrete "jagged" representations are superimposed to form a smooth, continuous curve.  

This allows the model to reconstruct high-precision "meaning-distances" through wave interference, much like how a Fourier series reconstructs smooth waves from simple components. It enables handheld devices to run large-scale models with the perceptual accuracy of high-bit floating-point math.  

---

#### License：  
This project is licensed under the Apache License 2.0. (See the LICENSE for details).  

#### Acknowledgments：  
This work builds upon the foundation established by the Transformer architecture.  
I would like to express my gratitude to the researchers and open-source communities
whose contributions to attention mechanisms, positional encoding, and large-scale
model design made this work possible.  
Neocognitron ― Transformer ― D‑RNA  Dream Resonance Never Adjourns — it goes on...  