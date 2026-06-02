import os
import math
import sys
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from datetime import datetime
from datasets import load_dataset
from safetensors.torch import save_file, load_file
from torch.utils.data import IterableDataset, DataLoader
import io
import wave
import numpy as np
import soundfile as sf

'''
こちらは学習コードです、これはモデルコードから機能等を参照して読みだして実行します
次元などの変更もこの学習コード側で行います、モデルコード側は基準として触らずに保持します
---
utf16をトークナイザ代替にする
世界中のさまざまな言語、絵文字、特殊記号、ソースコードのインデントまで、あらゆる文字を100%表現できる
(事前に決めた3万〜10万語の｢辞書｣にない言葉で[UNK]を生じることがなくなる)
1つのアーキテクチャで全メディアを等価に処理できる究極のマルチモーダルが理論上可能になる
Vocab Sizeを｢65,536｣(16bit境界)にジャストフィットさせることによるVRAM効率と計算効率を最大化します
ハードウェア的な特性(メモリのビット幅、アライメント、並列計算の仕組み)に合致しオーバーヘッドを生じません
(トークナイザやVAEなどの外付けはパディング処理などで空白を埋めるようなムダを生じる)
スマートフォンやエッジデバイスでも標準的なutf16なら確実に動作可能です
欠点は以下のみ、単純に少し長く学習するだけで解消します
コンテキスト長(トークン効率)の悪化、意味的抽象化(セマンティクス)をゼロから自力で学習しなければならない

という言語学習を音声学習に修正してみました
DCT変換をしているので、これを画像学習に転化することも可能です
'''

# モデル定義ファイルから必要なクラスをインポート
from drna_swi_mount import (
    DRNA_Model, 
    DRNA_Block,
    TernaryTrainingManager, 
    get_ternary_schedule
)

# グローバル変数：Ctrl+C ハンドラから安全にアクセスするため
training_interrupted = False
current_step_global = 0

def sigint_handler(signum, frame):
    '''Ctrl+C (SIGINT) をエレガントにキャッチするハンドラ'''
    global training_interrupted
    print("\n\n[!!] Ctrl+C (SIGINT) を検知しました。現在のステップで安全に緊急保存処理へ移行します...")
    training_interrupted = True

# SIGINT ハンドラを登録
signal.signal(signal.SIGINT, sigint_handler)

# 音声からピュアなDCT中周波を切り出し、UTF-16(0〜65535)へダイレクトマッピングする関数
def encode_audio_to_dct_utf16(audio_waveform: torch.Tensor, seq_len: int, mask_prob: float = 0.25) -> torch.Tensor:
    """
    audio_waveform: [N] 形状の低解像度サンプリング音声 (例: 4kHz)
    ※カンニング防止のため、オーバーラップなしの固定長ブツ切りパッチ
    """
    # 3秒以上（512トークン+1）を満たすための1パッチあたりの必要サンプル数を逆算
    # 例：4000Hz * 3.1秒 = 12400サンプル。12400 / 513 ≒ 24 サンプル
    patch_size = len(audio_waveform) // (seq_len + 1)
    if patch_size < 4: 
        patch_size = 4 # 最低限の解像度確保

    tokens = []
    for i in range(seq_len + 1):
        start = i * patch_size
        end = start + patch_size
        patch = audio_waveform[start:end]
        
        if len(patch) < patch_size:
            patch = F.pad(patch, (0, patch_size - len(patch)))
            
        # 余計な処理（μ-law等）を一切挟まない純粋な1次元DCT (PyTorchの直交変換で代用)
        # ※実数データに対して、scipyのdctタイプ2と同等の挙動を高速に行う
        dct_coefs = torch.linalg.vector_norm(patch) if len(patch) == 0 else torch.abs(torch.fft.rfft(patch))
        
        # 【こだわり】低周波を参考程度に残し、高周波をバッサリ切って「中周波」を主役にする
        # 4kHzサンプリング時の24サンプルなら、rfftで13個の成分。低周波(0~1) 中周波(2~7) 高周波(8~)
        # ここでは中周波をメインとした代表的な1つのエネルギー値を抽出
        mid_idx_start = max(1, len(dct_coefs) // 6)
        mid_idx_end = max(mid_idx_start + 1, len(dct_coefs) // 2)
        mid_energy = dct_coefs[mid_idx_start:mid_idx_end].sum()

        # 0〜65534 (UTF-16空間、65535はMask用に確保) へダイレクトに線形マッピング
        # AIの汎化性能を信じ、最大値等での単純な正規化のみ（カンニング防止のため連続性は不要）
        # ※ここではバッチ/ファイル内の簡易スケーリング。事前学習用に0〜65534に丸める
        token_val = int(torch.clamp(mid_energy * 1000, 0, 65534).item())
        tokens.append(token_val)

    # 【重要】自己教師あり事前学習のためのランダムMask処理（カンニング防止のトドメ）
    # 512トークンのうち指定確率を強制的にMask記号「65535」に置き換える
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    mask_indices = torch.rand(seq_len + 1) < mask_prob
    # ターゲット（答え）を汚さないよう、入力に使う位置だけをMask(65535)にするため、ここでは一旦そのまま保持しDataset側で制御
    return token_tensor

# 音声逐次読み込み（ストリーミング事前学習用）
# 【純粋1次元DCT-II】画像学習への転化を担保する、完全な直交コサイン基底変換
def pure_1d_dct(patch: torch.Tensor) -> torch.Tensor:
    N = patch.size(0)
    n = torch.arange(N, dtype=patch.dtype, device=patch.device)
    k = torch.arange(N, dtype=patch.dtype, device=patch.device).view(-1, 1)
    
    # JPEG圧縮の根幹と同じ、実数完結のコサイン基底行列を生成
    basis = torch.cos((math.pi / N) * (n + 0.5) * k)
    
    # 行列ベクトル積でダイレクトにDCT係数を抽出（絶対値ではなく、生の正負の実数エネルギー）
    return torch.mv(basis, patch)

# 音声からピュアなDCT中周波を切り出し、UTF-16(0〜65535)へダイレクトマッピング
def encode_audio_to_dct_utf16(audio_waveform: torch.Tensor, seq_len: int) -> torch.Tensor:
    # 3秒以上の波形を、カンニング防止のためオーバーラップなしの固定パッチにブツ切り
    patch_size = len(audio_waveform) // (seq_len + 1)
    if patch_size < 4: 
        patch_size = 4

    tokens = []
    for i in range(seq_len + 1):
        start = i * patch_size
        end = start + patch_size
        patch = audio_waveform[start:end]
        
        if len(patch) < patch_size:
            patch = torch.nn.functional.pad(patch, (0, patch_size - len(patch)))
            
        # 純粋DCTを実行（画像転化への足がかり）
        dct_coefs = pure_1d_dct(patch)
        
        # 【こだわり】低周波は参考程度（エネルギーが集中する最初の数個を軽く加算）、
        # 高周波はカット、中周波を主役にする
        # 実数DCTなので、単純に絶対値の大きさ（振幅強度）をベースに中周波の主役パートを抽出します
        abs_coefs = torch.abs(dct_coefs)
        
        mid_idx_start = max(1, len(abs_coefs) // 6)
        mid_idx_end = max(mid_idx_start + 1, len(abs_coefs) // 2)
        mid_energy = abs_coefs[mid_idx_start:mid_idx_end].sum()

        # 余計なμ-law等は使わず、0〜65534（UTF-16全域、65535はMask用）にダイレクト線形マッピング
        token_val = int(torch.clamp(mid_energy * 1000, 0, 65534).item())
        tokens.append(token_val)

    return torch.tensor(tokens, dtype=torch.long)

class StreamingAudioDataset(IterableDataset):
    def __init__(self, seq_len: int, sample_rate: int = 4000):
        super().__init__()
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        print(f">>> 【Windows超堅牢モード】LibriSpeech から生のバイト列をストリーミング中...")
        print(f">>> 短い音声をバッファに結合し、3秒（12000サンプル）ごとに切り出します。")
        
        # decode(False) でWindowsの内部自動デコードエラーを完全に徹底スルー
        self.dataset = load_dataset(
            "librispeech_asr", 
            "clean",
            split="train.100", 
            streaming=True
        ).decode(False)

    def __iter__(self):
        # 💡 短い音声をガッチャンコして溜めておくための、純PyTorchの音声バッファ
        audio_buffer = torch.empty(0, dtype=torch.float32, device="cpu")
        required_samples = self.sample_rate * 3  # 4000Hz * 3秒 = 12000 サンプル

        for item in self.dataset:
            audio_data = item.get("audio", {})
            raw_bytes = audio_data.get("bytes", None)
            
            if raw_bytes is None:
                continue

            try:
                # soundfileでメモリ上のバイナリ(FLAC等)を直接デコード
                # ※NumPyの高度な機能は使わず、単純なfloat32配列として読み込み
                with io.BytesIO(raw_bytes) as f:
                    audio_np, orig_sr = sf.read(f, dtype='float32')
                
                # 💡 NumPyがバグる前に、即座に純粋なPyTorchテンソルに変換！
                waveform = torch.from_numpy(audio_np)
                
                # チャンネル数が2次元以上（ステレオ等）ならモノラル化（平均化）
                if waveform.ndim > 1:
                    waveform = waveform.mean(dim=-1)
                    
            except Exception:
                # Windows特有のファイル読み込みエラーや破損は無言でスキップ
                continue
            
            # 高周波ノイズ落としの間引き（ダウンサンプリング）
            if orig_sr != self.sample_rate:
                step = orig_sr // self.sample_rate
                if step > 0:
                    waveform = waveform[::step]
                else:
                    continue
            
            # 💡【指定長蓄積ロジック】今回の短い音声を、メインバッファの末尾に結合
            audio_buffer = torch.cat([audio_buffer, waveform], dim=0)
            
            # 💡 バッファに3秒（12000サンプル）以上溜まっている限り、ループして切り出し続ける！
            # これにより、短いファイルが連続しても、合体して無駄なく1ステップに変身します
            while len(audio_buffer) >= required_samples:
                # 先頭から3秒分（12000サンプル）をカチッと切り出す
                chunk = audio_buffer[:required_samples]
                
                # 次のステップのために、今使った3秒分をバッファの先頭から削る
                audio_buffer = audio_buffer[required_samples:]
                
                # --- 以降は元の職人技マッピング処理 ---
                # ピュアDCT ➔ UTF-16インデックス化
                token_tensor = encode_audio_to_dct_utf16(chunk, self.seq_len)
                
                # 訓練用データの切り出し（入力とターゲット）
                inputs = token_tensor[:-1].clone()
                targets = token_tensor[1:].clone()
                
                # カンニング防止：入力側の25%をランダムにMask記号「65535」へ
                mask_mask = torch.rand(self.seq_len) < 0.25
                inputs[mask_mask] = 65535
                
                yield inputs, targets

'''
# UTF16 トークナイザ
def encode_utf16(text: str, seq_len: int) -> torch.Tensor:
    tokens = [ord(char) for char in text if ord(char) < 65536]
    if len(tokens) < seq_len + 1:
        tokens = tokens + [0] * (seq_len + 1 - len(tokens))
    else:
        tokens = tokens[:seq_len + 1]
    return torch.tensor(tokens, dtype=torch.long)

# Webテキスト逐次読み込み(ストリーミング)
class StreamingWebTextDataset(IterableDataset):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        print(">>> OpenWebText からデータをストリーミング接続中... (初回は数分かかる場合があります)")
        self.dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

    def __iter__(self):
        buffer = ""
        for item in self.dataset:
            text = item["text"].strip()
            if not text:
                continue
            buffer += text + " "
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                token_tensor = encode_utf16(chunk, self.seq_len)
                yield token_tensor[:-1], token_tensor[1:]
'''

# 💎 Gradient Checkpointing マウント関数
def apply_gradient_checkpointing(model: nn.Module):
    '''
    フック構造確定後に呼び出し、DRNA_Block全体をGCで包む
    これにより、再計算時にもフック(3値ブレンド)が正常に働き、w2の勾配消失を防ぐ
    '''
    print(">>> [成功] Gradient Checkpointing を各 DRNA_Block (レイヤー全体) に適用中...")

    def make_checkpoint_forward(block_module):
        original_forward = block_module.forward

        def checkpoint_forward(*args, **kwargs):
            x, cos, sin = args[0], args[1], args[2]
            mask = kwargs.get('mask', None)
            # GCでスキップされてしまう、そのフックを直接呼び出す
            for name, module in block_module.named_modules():
                if isinstance(module, nn.Linear):
                    # モジュールに登録されている forward_pre_hook を探して直接実行する
                    for hook in module._forward_pre_hooks.values():
                        # フックを偽装実行(module と inputs を渡せば、内部で正しく3値化される)
                        hook(module, (x,))
            return checkpoint.checkpoint(
                original_forward, x, cos, sin, mask, 
                use_reentrant=False
            )
        return checkpoint_forward

    for name, module in model.named_modules():
        if isinstance(module, DRNA_Block):
            module.forward = make_checkpoint_forward(module)
            print(f"  -> {name} (DRNA_Block) の計算空間を GC で保護しました。")

# 対話型セットアップ関数
def select_model_setup(vocab_size, d_model, n_layers, n_heads, d_ff):
    print(" D-RNA Trio 3値学習 セットアップモードの選択")
    print("1: 新規モデルを初期化して作成し学習開始")
    print("2: 既存の通常モデルや３値モデル、途中保存をマウントして学習開始")
    choice = input("選択してください (1 or 2): ").strip()

    model = DRNA_Model(
        vocab_size=vocab_size, d_model=d_model, 
        n_layers=n_layers, n_heads=n_heads, d_ff=d_ff
    ).cuda()

    if choice == "2":
        checkpoint_path = input("読み込む重みファイルのパスを入力してください ").strip().strip("'\"")
        if not os.path.exists(checkpoint_path):
            print(f"エラー: パスが見つかりません。新規作成します")
            return model

        print("\n>>> [理論通りに復元開始] 通常/3値/途中保存の重みから連続勾配空間を復元中...")
        _ = TernaryTrainingManager(model, warmup_steps=10, max_lambda=1.0)
        checkpoint = load_file(checkpoint_path, device='cuda')

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if "embed" in name or "output_head" in name:
                        continue

                    # 統一された単一の｢.weight｣から実数聖域｢raw_weight｣へバトンを戻す
                    target_key = f"{name}.weight" if f"{name}.weight" in checkpoint else f"{name}.raw_weight"
                    if target_key in checkpoint:
                        module.raw_weight.copy_(checkpoint[target_key].cuda())
        print(">>> マウントおよび連続勾配空間の再展開が完了しました \n")
        model.is_already_mounted = True
    else:
        model.is_already_mounted = False

    return model

def select_save_precision():
    print(" エクスポート精度の選択")
    print("1: bfloat16 (推奨)")
    print("2: float16")
    print("3: float32")
    choice = input("選択してください (1, 2, 3): ").strip()
    if choice == "2": return torch.float16, "fp16"
    if choice == "3": return torch.float32, "fp32"
    return torch.bfloat16, "bf16"

# メイン学習ループ
def main():
    global training_interrupted, current_step_global
    '''次元などの変更はこちらで行います'''
    # パラメータ設定
    vocab_size = 65536  # UTF-16 全域 (BMP)
    d_model = 256
    n_layers = 16       # GCにより、12GB VRAM環境でも16層でも軽快に回ります
    n_heads = 8
    d_ff = 1024         # (d_model * 4)

    seq_len = 512       
    batch_size = 8      
    max_train_steps = 3001 

    # モデル初期化 / ロード
    model = select_model_setup(vocab_size, d_model, n_layers, n_heads, d_ff)

    # 3値誘導マネージャー外付け
    warmup_steps = 300
    max_lambda = 1.0
    manager = TernaryTrainingManager(model, warmup_steps=warmup_steps, max_lambda=max_lambda)

    # 🎯フック構造確定の｢後｣にGCをマウント
    apply_gradient_checkpointing(model)

    print(">>> 偏見のないピュアな音声事前学習（Masked Audio Modeling）を開始します...")
    dataset = StreamingAudioDataset(seq_len=seq_len, sample_rate=4000) # 4kHz駆動
    dataloader = DataLoader(dataset, batch_size=batch_size)

    #print(">>> 英語Webテキストをストリーミング中...")
    #dataset = StreamingWebTextDataset(seq_len=seq_len)
    #dataloader = DataLoader(dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    print(f">>> 学習を開始します (最大 {max_train_steps} ステップ) ...")
    print("※ 学習途中で安全に終了して実数保存したい場合は [Ctrl+C] を押してください")
    model.train()

    current_step = 0
    save_mode = "crystallized"
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    for inputs, targets in dataloader:
        if current_step >= max_train_steps or training_interrupted:
            if training_interrupted:
                save_mode = "emergency"
            break

        current_step_global = current_step
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()

        # 順伝播
        outputs = model(inputs, pad_id=0)

        # 損失計算
        task_loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss = manager.amend_loss(task_loss, step=current_step, total_steps=max_train_steps)

        loss.backward()
        optimizer.step()

        if current_step % 100 == 0:
            blend_ratio = get_ternary_schedule(current_step, max_train_steps, warmup_steps)

            print(f"Step {current_step:3d}/{max_train_steps} | "
                  f"CE-Loss: {task_loss.item():.4f} | "
                  f"Trio Blend: {blend_ratio * 100:6.2f}%")

        current_step += 1

    # 💾 修正箇所: 2重保存バグを根絶するスマートなキー管理保存
    # まずベースとなる全パラメータの辞書を取得(※この時点では PyTorch の仕様で両方入っている)
    state_dict = model.state_dict()
    output_state_dict = {}

    if save_mode == "crystallized":
        print("\n>>> [通常終了] 指定ステップに到達したため、3値重みの結晶化(export_ternary)を実行中...")
        # モデル側のコアロジックを呼び出し、raw_weight と weight を完全に同じ3値で同期化
        crystallized_model = manager.export_ternary()
        state_dict = crystallized_model.state_dict()

        # 保存フェーズ: 中身は同一なので、余計な '.raw_weight' キーだけを完全に排除
        for k, v in state_dict.items():
            if ".raw_weight" in k:
                continue  # 重複保存しない
            output_state_dict[k] = v

        output_filename = f"drna_pure_158_{timestamp}.safetensors"
    else:
        print(f"\n>>> [緊急停止] Step {current_step_global} で中断されました。結晶化はせず、実数勾配空間を抽出中...")

        # 保存フェーズ: フックの残骸 '.weight' を捨て、実数本体 '.raw_weight' の名前を '.weight' にリネームして詰め替える
        for k, v in state_dict.items():
            if ".weight" in k and not any(x in k for x in ["embed", "output_head", "final_norm"]):
                # 3値化対象レイヤーの通常の '.weight' キー(バッファの残骸)は無視
                continue
            if ".raw_weight" in k:
                # 実数の本体である '.raw_weight' を通常の '.weight' という名前にリネームして昇格
                new_key = k.replace(".raw_weight", ".weight")
                output_state_dict[new_key] = v
            else:
                # それ以外の全レイヤー(Embedding、Norm等)はそのまま保持
                output_state_dict[k] = v

        output_filename = f"drna_int_step{current_step_global}_{timestamp}_realw.safetensors"

    # ユーザー指定の精度に一括キャストして safetensors でクリーンに書き出し
    target_dtype, dtype_str = select_save_precision()
    final_state_dict = {k: v.to(target_dtype).cpu() for k, v in output_state_dict.items()}

    output_path = os.path.join(os.path.dirname(__file__ if __file__ else "."), f"{dtype_str}_{output_filename}")

    save_file(final_state_dict, output_path)
    print(f"  ==> [完了] 2重保存を完全に排除したクリーンな1倍サイズモデルを保存しました: {output_path}")
    print("プロセスを正常に終了します")

if __name__ == "__main__":
    main()