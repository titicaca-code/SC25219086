import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import get_model
from utils import get_device, get_dataloaders, ensure_dir, load_checkpoint


@torch.no_grad()
def collect_predictions(model, dataloader, device, max_wrong_samples=25):
    model.eval()

    all_labels = []
    all_preds = []
    wrong_samples = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

        wrong_mask = preds != labels
        if wrong_mask.any():
            wrong_images = images[wrong_mask].detach().cpu()
            wrong_labels = labels[wrong_mask].detach().cpu()
            wrong_preds = preds[wrong_mask].detach().cpu()

            for img, true_label, pred_label in zip(wrong_images, wrong_labels, wrong_preds):
                if len(wrong_samples) < max_wrong_samples:
                    wrong_samples.append((img, int(true_label.item()), int(pred_label.item())))
                else:
                    break

        if len(wrong_samples) >= max_wrong_samples:
            # 继续收集全量预测用于混淆矩阵，但不再收集更多错分样本
            pass

    return np.array(all_labels), np.array(all_preds), wrong_samples


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("SVHN Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def unnormalize_image(img_tensor, mean, std):
    img = img_tensor.numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    return img


def plot_wrong_samples(wrong_samples, save_path, max_show=16):
    if len(wrong_samples) == 0:
        print("No wrong samples found.")
        return

    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]

    samples_to_show = wrong_samples[:max_show]
    n = len(samples_to_show)
    cols = 4
    rows = math.ceil(n / cols)

    plt.figure(figsize=(12, 3 * rows))

    for idx, (img, true_label, pred_label) in enumerate(samples_to_show, start=1):
        plt.subplot(rows, cols, idx)
        img_show = unnormalize_image(img, mean, std)
        plt.imshow(img_show)
        plt.title(f"True: {true_label} | Pred: {pred_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_dir = "./checkpoints"
    figure_dir = "./figures"
    ensure_dir(figure_dir)

    model_name = "enhanced"
    checkpoint_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _, test_dataset, _, test_loader = get_dataloaders(
        data_dir="./data/SVHN",
        batch_size=128,
        num_workers=0
    )

    model = get_model(model_name=model_name, num_classes=10).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)
    print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")
    print(f"Best test accuracy in checkpoint: {checkpoint['best_test_acc']:.4f}")
    print(f"Test dataset size: {len(test_dataset)}")

    y_true, y_pred, wrong_samples = collect_predictions(model, test_loader, device, max_wrong_samples=25)

    test_acc = (y_true == y_pred).mean()
    print(f"Recomputed Test Accuracy: {test_acc:.4f}")
    print(f"Collected wrong samples: {len(wrong_samples)}")

    cm_path = os.path.join(figure_dir, "confusion_matrix.png")
    wrong_path = os.path.join(figure_dir, "wrong_samples.png")

    plot_confusion_matrix(y_true, y_pred, cm_path)
    plot_wrong_samples(wrong_samples, wrong_path, max_show=16)

    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Wrong sample figure saved to: {wrong_path}")


if __name__ == "__main__":
    main()