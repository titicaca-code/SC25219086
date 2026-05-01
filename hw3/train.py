import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import PoetryLSTM
from utils import load_poems, load_vocab, PoetryDataset, collate_fn

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _ = model(x)  # [B, T, V]
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def plot_loss_curve(loss_list, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 路径
    data_dir = "./data"
    checkpoint_dir = "./checkpoints"
    figure_dir = "./figures"
    ensure_dir(checkpoint_dir)
    ensure_dir(figure_dir)

    # 超参数
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-3
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3

    # 数据
    poems = load_poems(os.path.join(data_dir, "qijue.txt"))
    vocab, char2idx, idx2char = load_vocab(data_dir)

    dataset = PoetryDataset(poems, char2idx)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"诗歌数量: {len(poems)}")
    print(f"词表大小: {len(vocab)}")
    print(f"batch 数量: {len(dataloader)}")

    # 模型
    model = PoetryLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    best_loss = float("inf")
    best_model_path = os.path.join(checkpoint_dir, "best_poetry_lstm.pth")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        loss_history.append(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "config": {
                        "vocab_size": len(vocab),
                        "embed_dim": embed_dim,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "dropout": dropout
                    }
                },
                best_model_path
            )

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch}/{num_epochs}] | Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time

    # 保存 loss 历史
    with open(os.path.join(figure_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)

    plot_loss_curve(loss_history, os.path.join(figure_dir, "loss_curve.png"))

    print("\nTraining finished.")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Total Training Time: {total_time / 60:.2f} min")
    print(f"Best model saved to: {best_model_path}")
    print(f"Loss curve saved to: {os.path.join(figure_dir, 'loss_curve.png')}")


if __name__ == "__main__":
    main()