import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model import get_model
from utils import (
    set_seed,
    get_device,
    get_dataloaders,
    calculate_accuracy,
    ensure_dir,
    save_checkpoint,
    plot_curves
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        _, preds = torch.max(outputs, dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        _, preds = torch.max(outputs, dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc


def main():
    # =========================
    # 1. 基本配置
    # =========================
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 你后面可以改这里的参数
    model_name = "enhanced"      # "simple" 或 "enhanced"
    data_dir = "./data/SVHN"
    batch_size = 128
    num_workers = 0              # Windows 下先设 0，最稳
    num_epochs = 15
    learning_rate = 1e-3
    weight_decay = 1e-4

    checkpoint_dir = "./checkpoints"
    figure_dir = "./figures"
    ensure_dir(checkpoint_dir)
    ensure_dir(figure_dir)

    # =========================
    # 2. 数据
    # =========================
    train_dataset, test_dataset, train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # =========================
    # 3. 模型、损失函数、优化器、调度器
    # =========================
    model = get_model(model_name=model_name, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    # =========================
    # 4. 训练记录
    # =========================
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": []
    }

    best_test_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")

    # =========================
    # 5. 训练循环
    # =========================
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_test_acc": best_test_acc,
                    "history": history
                },
                best_model_path
            )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - start_time

    print("\nTraining finished.")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Total Training Time: {total_time / 60:.2f} min")

    # =========================
    # 6. 保存最终训练曲线
    # =========================
    plot_curves(history, save_dir=figure_dir)
    print(f"Curves saved to: {figure_dir}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()