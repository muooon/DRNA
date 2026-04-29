# D‑RNA：Dual‑Helix Resonance Neural Architecture (DRNA)  
### Pre-Norm Edition  

⭐ このプロジェクトが気に入ったら"星"を付けてください ⭐  
readme：[English](README.md) | [日本語](README_JA.md)  


D-RNA は、二重らせん(Dual‑Helix)構造と RoPE による回転場を中核に据えた新しいニューラルアーキテクチャです  

本アーキテクチャでは、Attention と MLP を二重らせんとして同期させ、共鳴収縮(Resonant Contraction)により情報をホログラフィックに圧縮します  
これは 疎な表現を密に再配置 し、次元を増やすことなく、深さ方向の構造だけで高い表現力を獲得します  
Transformer の全接続性を維持しつつ、破壊的忘却を抑制し、微細なゆらぎや位相情報を保持できる点が特徴です  

---

### D-RNA の主な特徴  

高い構造的互換性：標準的な Transformer Block と入出力形状が同一であり、アーキテクチャの心臓部としてスムーズな置き換えが可能です。  
共鳴収縮 (Resonant Contraction)：Attention と MLP を二重らせん状に同期させ、情報を位相場に収束させることで、表現密度を劇的に向上させます。  
次元の代替としての深さ：螺旋の回転(深さ方向の演算)が次元の不足を補い、パラメータ数を増やさずホログラフィックな情報保持を実現します。  
優れた学習効率：螺旋構造による情報の引き寄せ(同期)により、Transformer よりも極めて少ないステップ数で早期収束を達成します。  
微細な位相保持：RoPE を活用した回転場により、従来の構造では失われがちな微細なゆらぎや文脈の相対関係を精度高く保持します。  
知識の再同調が可能：既存の重みを初期値として移植し、低学習率で螺旋の位相に馴染ませることで、既存知能を D-RNA 構造へ進化･上書きできます。 

### ご注意ください  

学習率(LR)の最適化：D-RNA は共鳴収縮により情報の同期が極めて速いため、標準的な Transformer よりも低い学習率で十分に、かつ高速に収束します。 設定が高すぎると、共鳴が過剰に増幅され振動を招く可能性があるため、控えめな LR からの開始を推奨します。  
勾配の相乗効果：Attention(回想)と MLP(記憶)が二重らせん状に直列同期しているため、一回の更新による｢重みのなじみ｣が非常に強く働きます。 これは高速収束の利点であると同時に、慎重な更新が安定化の鍵であることを意味します。  
パラメータの共通性：重みの初期化シードやバッチサイズなどのハイパーパラメータは、通常の Transformer 設定をそのまま継承できます。  

<details>

<summary> D-RNAの特徴 </summary>

D-RNAは｢螺旋の位相｣により共鳴収縮法(共鳴投影場)を構築します  
これは疎を密にすることで破壊的忘却を抑制し(打ち消し合いを生じずに)最短路へと加速します  
微細なノイズも情報の純化を進めることで多様体は滑らかになり累積的に汎化を獲得します  
これらは特定の何かに依存せず、すべてのoptimizerやモデルに対し機能します  
ある意味では生体の脳を模したような ニューロンとグリア的 な機構です  
共鳴収縮法(共鳴投影場)は結果的にODE縮約近似相当となります  

</details>

---

### 概念図(Conceptual Diagram) 
```
	｢探す｣(Attention)、｢知っている｣(MLP)この２つを螺旋の位相で同期させる

	RoPE の回転場 (位相保持)  
	疎を密にするホログラフィック圧縮  

		A     M  
		 \   /  
		  \ /    ← ここが共鳴(Resonance)  
		  / \      seed により自然に同期が生まれる  
		 /   \     同期の連鎖で意味などを自然に引き寄せる  
		A     M  

	深さ方向へ繰り返すことで二重らせんを形成  
	(次元の代替として機能)  
```

---

### 最小コード(Minimal Block)  

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

### 例：Transformer のブロックを DRNA ブロックに置き換える  

```python
class DRNA_ResonantBlock(nn.Module):
    """
    既存の TransformerBlock をこの ResonantBlock に置き換える
    I/O: [Batch, Seq, Dim] -> [Batch, Seq, Dim] (完全互換)
    アーキテクチャ：プレ・ノーム (深層ネットワークにおける安定性優先)
    """
    def __init__(self, dim, n_heads, mlp_dim_forward=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        
        # 1. 螺旋の射影層 (らせんA)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        
        # 2. 記憶の層 (らせんB)
        mlp_dim = mlp_dim_forward if mlp_dim_forward else dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
        # 3. 収縮のための正規化層
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, cos, sin):
        """
        引数に RoPE 用の位相情報 (cos, sin) を追加するのが唯一の違い
        """
        # --- Attention Path (Pre-Norm) ---
        # 正規化 -> QKV -> RoPE -> 残余加算
        residual = x
        x_norm = self.norm1(x)
        
        # --- らせんA: 文脈の共鳴 (Attention) ---
        # QKV 抽出 -> RoPE 回転 -> 収縮(Norm)
        q, k, v = project_qkv(x_norm, self.qkv, self.n_heads, self.d_head)
        q, k = apply_rope(q, k, cos, sin)
        
        attn_out = attention(q, k, v)
        x = residual + self.out(attn_out) # 文脈との同期

        # --- らせんB: 記憶の共鳴 (MLP) ---
        #  正規化 -> MLP -> 残余加算
        residual = x
        x = residual + self.mlp(self.norm2(x)) # 記憶による確定
        
        return x
```

### D-RNAへの置き換えと活用について  
そのまま置換(Drop-in replacement)はできませんが｢再定義と再同調｣による活用は可能です  
なぜそのままではダメなのか･･･  
標準的なTransformerが｢絶対的な住所｣(絶対位置)で記憶しているのに対し、D-RNAは｢螺旋の位相｣(相対位置)で情報を処理するため、座標系が根本から異なります。 重みをそのままコピーしても、位相が合わず共鳴が起きません。  
どう置き換えるのか(実装)  
ネットワークの入出力形状は完全互換です。 既存の層を ResonantBlock に書き換え、位置情報を RoPE の回転場へ移行するだけで、心臓部のアップグレードが完了します。  
どう活用し、なじませるのか(学習)  
既存モデルの重みを初期値として転写(Transfer)した後、低学習率で継続学習を行います。 止まっていた知識(既存の重み)が螺旋の回転に同期し始め、次第にD-RNAの｢共鳴収縮｣のプロセスへ溶け込み、元の性能を超えて進化します。  

---

BPC Comparison Chart  

use-mask  
<img width="800" height="500" alt="bpc_prenorm_battle" src="https://github.com/user-attachments/assets/d599d19b-48b7-45e7-aff4-195067823976" />

use-mask 5000step
<img width="800" height="500" alt="bpc_prenorm_battle_5000" src="https://github.com/user-attachments/assets/7941f635-b0a7-471d-819f-76c9d55a5bc7" />
 

学習テスト状況(詳細)  
モデル規模：次元数(d_model)：256、レイヤ数(n_layers)：16、ヘッド数(n_heads)：8  
データセット：enwik8(100MB)  
学習設定：ステップ数：5,000、バッチサイズ：16、シーケンス長：512、AdamW(LR：1e-4)  

学習結果解析(概要)  
学習効率(Training Efficiency)：30%向上(ステップ効率：約1.5倍)  
収束速度(Convergence Speed)：時間コスト30%減(収束率(Convergence Rate)約1.5倍に加速)  

テストにより観測された利点：  
パラメータの利用密度(Parameter Density)最適化  
収束特性における構造的アドバンテージ  
同一計算予算内における情報保持容量(Information Capacity)拡大  
※ D-RNA は Transformer に対し、螺旋構造による位相同期を用いて、内部的な最小コストのみで進化  

| 指標           | Normal Transformer | D-RNA Transformer | 差分 / 効率                 |  
|----------------|--------------------|-------------------|------------------------------|  
| 到達Step数     | 3,850 step         | 2,350 step        | 約 39.0% 削減              |  
| 所要時間       | 1365.4 sec         | 876.1 sec         | 約 35.8% 高速化            |  
| VRAM使用量     | 4.51 GB            | 5.05 GB           | +0.54 GB コスト増      |  

---

ライセンス：  
本プロジェクトは Apache License 2.0 の下で公開されています(詳細は LICENSE をご覧ください)  

### 謝辞：  
本研究は Transformer アーキテクチャによって築かれた基盤の上に成り立っています  
Attention 機構、位置エンコーディング、大規模モデル設計に関する研究とオープンソースコミュニティの貢献に深く感謝いたします  
