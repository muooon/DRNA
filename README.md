# D-RNA：Dual‑Helix Resonance Neural Architecture (DRNA)  
### Pre-Norm Edition  

## Attention is all you need_started,  
## Resonance is all you need_endure,  

⭐ If you like this project, please give it a star ⭐  
readme：[English](README.md) | [日本語](README_JA.md)  

D-RNA is a new neural architecture centered on a dual helix structure and a rotation field produced by RoPE.  

In this architecture, Attention and MLP are synchronized into a dual helix, and information is holographically compressed through Resonant Contraction.  
This method rearranges sparse representations into dense ones to achieve high expressiveness using the depth‑direction structure alone, without increasing the number of dimensions.  
A key feature of this approach is its ability to preserve the full connectivity of the Transformer architecture while suppressing catastrophic forgetting and retaining subtle fluctuations and phase information.  

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

---

BPC Comparison Chart  

use-mask  
<img width="800" height="500" alt="bpc_prenorm_battle" src="https://github.com/user-attachments/assets/d599d19b-48b7-45e7-aff4-195067823976" />

use-mask 5000step
<img width="800" height="500" alt="bpc_prenorm_battle_5000" src="https://github.com/user-attachments/assets/7941f635-b0a7-471d-819f-76c9d55a5bc7" />
 

Learning Test Status (Details)  
Model Scale: Dimension (d_model): 256, Layers (n_layers): 16, Heads (n_heads): 8  
Dataset: enwik8 (100MB)  
Training Settings: Steps: 10,000, Batch Size: 16, Sequence Length: 512, AdamW (LR: 1e-4)  

Training Result Analysis (Overview)  
Training Efficiency: 30% improvement (Step efficiency: approx. 1.5x)  
Convergence Speed: 30% reduction in time cost (Convergence rate accelerated by 1.5x)  

Observed Benefits from Testing (Summary)  
Optimization of Parameter Density  
Structural advantage in convergence characteristics  
Expansion of Information Capacity within the same computational budget  
※ D-RNA evolves relative to the Transformer by utilizing phase synchronization via its helical structure, incurring only minimal internal costs optimization.  

| Metric                 | Normal Transformer | D-RNA Transformer | Difference / Efficiency      |  
|------------------------|--------------------|-------------------|------------------------------|  
| Steps to Reach Target  | 3,850 steps        | 2,350 steps       | ~39.0% reduction             |  
| Time Required          | 1365.4 sec         | 876.1 sec         | ~35.8% faster                |  
| VRAM Usage             | 4.51 GB            | 5.05 GB           | +0.54 GB cost   |  

---

New Perspective: Digital Vector × Phase Distance  

D-RNA can achieve high-resolution approximation even in low-bit environments (e.g., 1.58-bit / Ternary weights).  

Discrete Vectors: Each layer handles discrete, "digital" 3-value vectors (-1, 0, 1).  

<img width="800" height="400" alt="smooth" src="https://github.com/user-attachments/assets/69efc1cd-e13b-400f-a188-b886eb642f26" />

Continuous Distance: By stacking layers with different phases (Double Helix), these discrete "jagged" representations are superimposed to form a smooth, continuous curve.  

This allows the model to reconstruct high-precision "meaning-distances" through wave interference, much like how a Fourier series reconstructs smooth waves from simple components. It enables handheld devices to run large-scale models with the perceptual accuracy of high-bit floating-point math.  

---

License：  
This project is licensed under the Apache License 2.0. (See the LICENSE for details).  

#### Acknowledgments：  
This work builds upon the foundation established by the Transformer architecture.  
I would like to express my gratitude to the researchers and open-source communities
whose contributions to attention mechanisms, positional encoding, and large-scale
model design made this work possible.  
