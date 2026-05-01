import os
import json
import torch
from torch.utils.data import Dataset


def load_poems(txt_path):
    poems = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                poems.append(line)
    return poems


def build_vocab(poems):
    """
    构建字符级词表
    增加两个特殊符号：
    <PAD> 用于补齐
    <UNK> 用于未知字符
    """
    chars = set()
    for poem in poems:
        chars.update(list(poem))

    chars = sorted(list(chars))

    vocab = ["<PAD>", "<UNK>"] + chars
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}

    return vocab, char2idx, idx2char


def encode_text(text, char2idx):
    unk_idx = char2idx["<UNK>"]
    return [char2idx.get(ch, unk_idx) for ch in text]


def decode_indices(indices, idx2char):
    chars = []
    for idx in indices:
        ch = idx2char.get(int(idx), "")
        if ch not in ["<PAD>", "<UNK>"]:
            chars.append(ch)
    return "".join(chars)


def save_vocab(save_dir, vocab, char2idx, idx2char):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "char2idx.json"), "w", encoding="utf-8") as f:
        json.dump(char2idx, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "idx2char.json"), "w", encoding="utf-8") as f:
        json.dump(idx2char, f, ensure_ascii=False, indent=2)


def load_vocab(save_dir):
    with open(os.path.join(save_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(os.path.join(save_dir, "char2idx.json"), "r", encoding="utf-8") as f:
        char2idx = json.load(f)

    with open(os.path.join(save_dir, "idx2char.json"), "r", encoding="utf-8") as f:
        idx2char_raw = json.load(f)

    idx2char = {int(k): v for k, v in idx2char_raw.items()}
    return vocab, char2idx, idx2char


class PoetryDataset(Dataset):
    """
    输入: 前 n-1 个字符
    标签: 后 n-1 个字符
    例如：
    ABCDE
    input = ABCD
    target = BCDE
    """
    def __init__(self, poems, char2idx):
        self.samples = []
        for poem in poems:
            encoded = encode_text(poem, char2idx)
            if len(encoded) >= 2:
                self.samples.append(encoded)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    """
    由于这里所有七言绝句长度固定为 28，
    所以理论上不需要 pad，但这里保留通用写法。
    """
    xs, ys = zip(*batch)

    max_len = max(x.size(0) for x in xs)

    padded_x = []
    padded_y = []

    for x, y in zip(xs, ys):
        pad_len = max_len - x.size(0)

        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            y = torch.cat([y, torch.zeros(pad_len, dtype=torch.long)], dim=0)

        padded_x.append(x)
        padded_y.append(y)

    return torch.stack(padded_x), torch.stack(padded_y)


if __name__ == "__main__":
    txt_path = "./data/qijue.txt"
    poems = load_poems(txt_path)

    print(f"诗歌数量: {len(poems)}")

    vocab, char2idx, idx2char = build_vocab(poems)
    print(f"词表大小: {len(vocab)}")

    print("\n前 3 首诗：")
    for i, poem in enumerate(poems[:3], start=1):
        print(f"{i}: {poem}")

    sample = poems[0]
    encoded = encode_text(sample, char2idx)
    decoded = decode_indices(encoded, idx2char)

    print("\n编码前样例:")
    print(sample)
    print("\n编码后前 20 个 index:")
    print(encoded[:20])
    print("\n解码还原:")
    print(decoded)

    save_vocab("./data", vocab, char2idx, idx2char)
    print("\n词表已保存到 ./data/")