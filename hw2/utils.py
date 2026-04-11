import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


def set_seed(seed=42):
    """
    固定随机种子，保证实验尽可能可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了结果更稳定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    自动选择设备
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_transforms():
    """
    返回基础版和增强版常用的数据预处理
    SVHN 是 RGB 彩色图像，尺寸 32x32
    """
    # 常用经验值，可直接用于 SVHN
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform


def get_dataloaders(data_dir="./data/SVHN", batch_size=128, num_workers=2):
    """
    加载 SVHN 数据集
    如果本地没有，会自动下载
    """
    train_transform, test_transform = get_data_transforms()

    train_dataset = datasets.SVHN(
        root=data_dir,
        split="train",
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.SVHN(
        root=data_dir,
        split="test",
        download=True,
        transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataset, test_dataset, train_loader, test_loader


def calculate_accuracy(outputs, labels):
    """
    计算一个 batch 的准确率
    """
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def ensure_dir(path):
    """
    如果文件夹不存在，则创建
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(state, filepath):
    """
    保存模型检查点
    """
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, map_location="cpu"):
    """
    加载模型检查点
    """
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def plot_curves(history, save_dir="./figures"):
    """
    绘制训练/测试损失曲线和准确率曲线
    history 是一个字典，例如：
    {
        "train_loss": [...],
        "test_loss": [...],
        "train_acc": [...],
        "test_acc": [...]
    }
    """
    ensure_dir(save_dir)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Accuracy 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Test Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close()


def imshow_tensor(img_tensor, mean=None, std=None):
    """
    用于可视化单张图片
    """
    img = img_tensor.detach().cpu().numpy().transpose((1, 2, 0))

    if mean is not None and std is not None:
        mean = np.array(mean)
        std = np.array(std)
        img = std * img + mean

    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis("off")


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Device:", device)

    train_dataset, test_dataset, train_loader, test_loader = get_dataloaders(
        data_dir="./data/SVHN",
        batch_size=64,
        num_workers=0
    )

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    print("Sample labels:", labels[:10].tolist())