import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import torch.nn.functional as F

from model import PoetryLSTM
from utils import load_vocab, decode_indices


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    model = PoetryLSTM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, checkpoint


def sample_next_char(logits, temperature=1.0, top_k=8):
    """
    从最后一个时间步的 logits 中采样下一个字符
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k is not None and top_k > 0:
        top_probs, top_indices = torch.topk(probs, k=top_k)
        top_probs = top_probs / top_probs.sum()
        next_idx = top_indices[torch.multinomial(top_probs, num_samples=1)]
        return next_idx.item()
    else:
        next_idx = torch.multinomial(probs, num_samples=1)
        return next_idx.item()


def generate_poem(
    model,
    char2idx,
    idx2char,
    start_text="明月",
    gen_len=28,
    temperature=0.85,
    top_k=10,
    device="cpu"
):
    """
    生成固定长度古诗
    """
    # 起始词转 index
    input_indices = []
    unk_idx = char2idx["<UNK>"]
    for ch in start_text:
        input_indices.append(char2idx.get(ch, unk_idx))

    generated = input_indices[:]

    hidden = None

    # 先把起始词喂进去
    x = torch.tensor([input_indices], dtype=torch.long).to(device)
    with torch.no_grad():
        logits, hidden = model(x, hidden)

    # 继续生成直到达到目标长度
    while len(generated) < gen_len:
        last_token = torch.tensor([[generated[-1]]], dtype=torch.long).to(device)

        with torch.no_grad():
            logits, hidden = model(last_token, hidden)

        next_logits = logits[0, -1, :]
        next_idx = sample_next_char(next_logits, temperature=temperature, top_k=top_k)
        generated.append(next_idx)

    poem = decode_indices(generated, idx2char)
    return poem[:gen_len]


def format_qijue(poem_text):
    """
    将 28 字字符串格式化为四句七言绝句
    """
    if len(poem_text) < 28:
        return poem_text

    lines = [
        poem_text[0:7],
        poem_text[7:14],
        poem_text[14:21],
        poem_text[21:28],
    ]
    return "，\n".join(lines[:3]) + "。\n" + lines[3] + "。"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = "./data"
    checkpoint_path = "./checkpoints/best_poetry_lstm.pth"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    vocab, char2idx, idx2char = load_vocab(data_dir)
    model, config, checkpoint = load_model(checkpoint_path, device)

    print(f"Loaded model from epoch: {checkpoint['epoch']}")
    print(f"Best loss: {checkpoint['best_loss']:.4f}")

    start_text = "明月"

    print("\n以“明月”为起始词生成 5 首七言绝句：\n")

    results = []
    for i in range(5):
        poem = generate_poem(
            model=model,
            char2idx=char2idx,
            idx2char=idx2char,
            start_text=start_text,
            gen_len=28,
            temperature=0.8,
            top_k=8,
            device=device
        )
        formatted = format_qijue(poem)

        print(f"【第 {i+1} 首】")
        print(formatted)
        print()

        results.append({
            "index": i + 1,
            "raw": poem,
            "formatted": formatted
        })

    with open(os.path.join(output_dir, "generated_poems.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("生成结果已保存到: ./outputs/generated_poems.json")


if __name__ == "__main__":
    main()