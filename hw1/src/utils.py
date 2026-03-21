import os
import matplotlib.pyplot as plt


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_true_vs_pred(y_true, y_pred, save_path, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_correlation_heatmap(df, save_path):
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()