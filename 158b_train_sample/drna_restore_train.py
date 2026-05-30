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

    seq_len = 256       
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

    print(">>> 英語Webテキストをストリーミング中...")
    dataset = StreamingWebTextDataset(seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

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