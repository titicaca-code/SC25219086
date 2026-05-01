import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import matplotlib.pyplot as plt


def main():
    loss_json_path = "./figures/loss_history.json"
    save_path = "./figures/loss_curve.png"

    with open(loss_json_path, "r", encoding="utf-8") as f:
        loss_history = json.load(f)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Loss curve saved to: {save_path}")


if __name__ == "__main__":
    main()