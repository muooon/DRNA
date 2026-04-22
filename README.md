# D-RNA：Dual‑Helix Resonance Neural Architecture (DRNA)  

[English](README.md) | [日本語](README.jp.md)

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
Since Attention (recall) and the MLP (memory) are synchronized in a double‑helix sequence, the “settling” of weights from a single update is very strong.  
This is an advantage for fast convergence, but it also means that careful updates are key to stability.  
Parameter commonality:  
Hyperparameters such as weight initialization seeds and batch size can be inherited directly from standard Transformer settings.  

---

### Conceptual Diagram

```
	Synchronizing “searching” (Attention) and “knowing” (MLP) in the phase of a spiral.

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
        # --- Attention ---
        q, k, v = project_qkv(x, self.qkv, self.n_heads, self.d_head)
        q, k = apply_rope(q, k, cos, sin)
        attn_out = attention(q, k, v)
        x = self.norm1(x + self.out(attn_out))

        # --- MLP ---
        x = self.norm2(x + self.mlp(x))
        return x
```

---

### Example: Replacing a Transformer block with a D-RNA block  

```python
class DRNA_ResonantBlock(nn.Module):
    """
    Replace the existing TransformerBlock with this ResonantBlock.
    I/O: [Batch, Seq, Dim] -> [Batch, Seq, Dim] (Fully compatible)
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
        
        # 3. Normalization layer for compression
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, cos, sin):
        """
        Phase information for RoPE as an argument (cos, sin)
        """
        # Attention：Spiral Projection Layer (A)
        # QKV -> RoPE -> Norm
        q, k, v = project_qkv(x, self.qkv, self.n_heads, self.d_head)
        q, k = apply_rope(q, k, cos, sin)
        
        attn_out = attention(q, k, v)
        x = self.norm1(x + self.out(attn_out)) # Synchronization with context

        # MLP：Spiral Memory Layer (B)
        # MLP -> Norm
        x = self.norm2(x + self.mlp(x)) # Determined by memory
        
        return x
```

### Replacement and Utilization of D-RNA  
A direct drop‑in replacement is not possible, but it can be utilized through “redefinition and re‑synchronization.”  
Why it cannot be used as‑is:  
While a standard Transformer stores information using an “absolute address” (absolute position), D-RNA processes information using the “phase of a spiral” (relative position), meaning the coordinate systems are fundamentally different.  
Even if the weights are copied directly, the phases do not align and no resonance occurs.  
How to replace it (implementation):  
The network’s input–output shapes are fully compatible.  
By rewriting the existing layers as ResonantBlock and migrating positional information into RoPE’s rotational field, the core upgrade is complete.  
How to utilize and adapt it (training):  
After transferring the existing model’s weights as initialization, continue training with a low learning rate.  
The previously static knowledge (existing weights) begins to synchronize with the spiral rotation, gradually blending into D-RNA’s “Resonant Contraction” process and evolving beyond the original performance.  

---

BPC Comparison Chart  

non-mask  
<img width="800" height="500" alt="bpc_only" src="https://github.com/user-attachments/assets/36a68185-ebf4-4518-83c0-ccb891d327a4" />

use-mask  
<img width="800" height="500" alt="bpc_mask" src="https://github.com/user-attachments/assets/3e54dcd9-b143-4fbe-a670-91005d2acecf" />

---

License：  
This project is licensed under the Apache License 2.0. (See the LICENSE for details).  

#### Acknowledgments：  
This work builds upon the foundation established by the Transformer architecture.  
I would like to express my gratitude to the researchers and open-source communities
whose contributions to attention mechanisms, positional encoding, and large-scale
model design made this work possible.  
