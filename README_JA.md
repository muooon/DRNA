# D‑RNA：Dual‑Helix Resonance Neural Architecture (DRNA)  
### Pre-Norm Edition  

### 必要なのは「注意」だけ (Attention is all you need_started,)  
### ｢共鳴｣で保ち続けるだけ (Resonance is all you need_endure,)  
### Neocognitron ― Transformer ― Dream Resonance Never Adjourns — it goes on...  

⭐ このプロジェクトが気に入ったら"星"を付けてください ⭐  
readme：[English](README.md) | [日本語](README_JA.md)  


D-RNA は、二重らせん(Dual‑Helix)構造と RoPE による回転場を中核に据えた新しいニューラルアーキテクチャです  

本アーキテクチャでは、Attention と MLP を二重らせんとして同期させ、共鳴収縮(Resonant Contraction)により情報をホログラフィックに圧縮します  
これは 疎な表現を密に再配置 し、次元を増やすことなく、深さ方向の構造だけで高い表現力を獲得します  
Transformer の全接続性を維持しつつ、破壊的忘却を抑制し、微細なゆらぎや位相情報を保持できる点が特徴です  

---

### 解説 ･ Explanation  
二重らせん共鳴収縮構造(D‑RNA)による Transformer の高密度化と高速収束  

#### [数学的解説はこちら(論文)](https://huggingface.co/muooon/DRNA/raw/main/D%E2%80%91RNA_paper_260507(JPN).txt)  

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

<details>

<summary> D-RNAの置き換えと活用 </summary>

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

</details>

---

## BPC Comparison Chart  

最新版テスト結果(適正学習可)  
<img width="800" height="500" alt="bpc_prenorm_battle_Lay16-4x07x5000" src="https://github.com/user-attachments/assets/7239f630-e890-4997-93ff-7495e5c7af81" />  

AdamW：5e-5  
Vanilla:16L, VRAM:2.11GB, Step 5000 | BPC: 3.2970 | 801.9s  
D-RNA:4L, VRAM:1.03GB, Step 5000 | BPC: 2.8744 | 316.4s  
|効率化| VRAM：約50％削減、 BPC：精度向上、 時間：約2.5倍高速化  

<details>

<summary> Kv-RoPE制限後のログ </summary>

学習テスト状況(詳細)：  
モデル規模：次元数(d_model)：128、レイヤ数(n_layers)：16/4(D-RNA)、ヘッド数(n_heads)：4  
データセット：enwik8(100MB)  
学習設定：ステップ数：5,000、バッチサイズ：16、シーケンス長：512、AdamW(LR：5e-5)  

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

<summary> Kv-RoPE制限前テスト結果 </summary>

||| Phase 1：純粋構造比較(同一層数：16L vs 16L) |||

use-mask  
<img width="800" height="500" alt="bpc_prenorm_battle" src="https://github.com/user-attachments/assets/d599d19b-48b7-45e7-aff4-195067823976" />

use-mask 5000step
<img width="800" height="500" alt="bpc_prenorm_battle_5000" src="https://github.com/user-attachments/assets/7941f635-b0a7-471d-819f-76c9d55a5bc7" />
 

学習テスト状況(詳細)：  
モデル規模：次元数(d_model)：256、レイヤ数(n_layers)：16、ヘッド数(n_heads)：8  
データセット：enwik8(100MB)  
学習設定：ステップ数：5,000、バッチサイズ：16、シーケンス長：512、AdamW(LR：1e-4)  

学習結果解析(概要)：  
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


||| Phase 2：最適化の実装(レイヤーの1/2化：16L vs 8L) |||

use-mask 5000step  
<img width="800" height="500" alt="bpc_prenorm_battle_5000xcos" src="https://github.com/user-attachments/assets/8a75106e-65c5-4e2b-b842-39d0b92a1a00" />

use-mask 10000step  
<img width="800" height="500" alt="bpc_prenorm_battle_10000xcosGeLU" src="https://github.com/user-attachments/assets/0e3d2f9e-5507-4ec2-9349-deb4d47cfbe5" />  

学習テスト状況(詳細)：  
モデル規模：次元数(d_model)：256、レイヤ数(n_layers)：16 / 8(D-RNA)、ヘッド数(n_heads)：8  
データセット：enwik8(100MB)  
学習設定：ステップ数：10,000、バッチサイズ：16、シーケンス長：512、AdamW(LR：1e-4)、CosineAnnealing  

学習結果解析(概要)：  
学習効率(Training Efficiency)：30%向上(ステップ効率：約1.5倍)  
※ 同一精度(BPC 2.05付近)への到達において、最大で 約60%の時間短縮 を観測  
収束速度(Convergence Speed)：時間コスト30%減(収束率(Convergence Rate)約1.5倍に加速)  
※ 物理レイヤ削減(16L → 8L)と螺旋構造(位相同期)で、パラメータ密度の最適化と高速化を両立  

テストにより観測された利点：  
パラメータの利用密度(Parameter Density)最適化  
収束特性における構造的アドバンテージ  
同一計算予算内における情報保持容量(Information Capacity)拡大  
※ D-RNA は Transformer に対し、螺旋構造による位相同期を用いて、内部的な最小コストのみで進化  

| 指標           | Normal Transformer 16 | D-RNA Transformer 8 | 差分 / 効率                 |  
|----------------|--------------------|-------------------|------------------------------|  
| 到達Step数     | 4,650 step         | 3,650 step        | 約 21.0% 削減              |  
| 所要時間       | 1641.5 sec         | 682.9 sec         | 約 58.0% 高速化            |  
| VRAM使用量     | 4.79 GB            | 2.56 GB           | 約 46% 削減      |  
| 最終BPC       | 1.9272          | 1.8958          | 同等(レイヤあたり精度向上)        |  
| 最終step時間     | 3,542.2 sec         | 1,876.8 sec        | 約 53.0% 短縮      |  


基本アーキテクチャ仕様：  
Transformer   Absolute(Learned)   GELU  
D-RNA   RoPE(Rotary)   GELU

</details>

---

新たな視点：デジタルベクトル × 位相距離  

D-RNAは、低ビット環境(例：1.58b / 3値重み)でも高解像度の近似を実現可能です  

離散ベクトル：各レイヤーは離散的な｢デジタル｣3値ベクトル(-1、0、1)を処理します  

<img width="800" height="400" alt="smooth" src="https://github.com/user-attachments/assets/69efc1cd-e13b-400f-a188-b886eb642f26" />

連続距離：異なる位相(二重らせん)を持つ層を積み重ねると、離散的な｢ギザギザ｣の階段状の表現が重ね合わされ、滑らかで連続的な曲線を形成します  

これは、フーリエ級数が単純な構成要素から滑らかな波を再構築するのと同様に、波の干渉を通して高精度の｢意味距離｣をモデルが再構築することが可能になります。 これは携帯端末でも、高ビット浮動小数点演算と同等の知覚精度で大規模モデルを実行できるようになり得る可能性を秘めています  

---

### ライセンス：  
本プロジェクトは Apache License 2.0 の下で公開されています(詳細は LICENSE をご覧ください)  

### 謝辞：  
本研究は Transformer アーキテクチャによって築かれた基盤の上に成立します  
Attention 機構、位置エンコーディング、大規模モデル設計に関する研究とオープンソースコミュニティの貢献に深く感謝いたします  
Neocognitron ― Transformer ― D-RNA 夢の共鳴は決して休まず続く･･･ (D-RNA Dream Resonance Never Adjourns — it goes on...)  