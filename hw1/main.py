import os

from src.data_process import load_and_split_data
from src.linear_regression import run_linear_regression
from src.mlp_regression import run_mlp_regression
from src.utils import ensure_dir, plot_true_vs_pred, plot_correlation_heatmap


def main():
    data_path = os.path.join("hw1", "data", "Concrete_Data_Yeh.csv")
    results_dir = os.path.join("hw1", "results")

    ensure_dir(results_dir)

    (
        df,
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_scaled,
        y_test_scaled,
        x_scaler,
        y_scaler,
    ) = load_and_split_data(data_path)

    print("Data shape:", df.shape)
    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])

    plot_correlation_heatmap(df, os.path.join(results_dir, "corr_heatmap.png"))

    lr_model, lr_pred, lr_mse = run_linear_regression(X_train, X_test, y_train, y_test)
    print(f"Linear Regression Test MSE: {lr_mse:.4f}")
    plot_true_vs_pred(
        y_test.flatten(),
        lr_pred.flatten(),
        os.path.join(results_dir, "linear_pred_vs_true.png"),
        "Linear Regression: True vs Predicted"
    )

    mlp_model, mlp_pred, mlp_mse = run_mlp_regression(
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_scaled,
        y_scaler,
        epochs=1000,
        lr=0.001
    )
    print(f"MLP Regression Test MSE: {mlp_mse:.4f}")
    plot_true_vs_pred(
        y_test.flatten(),
        mlp_pred.flatten(),
        os.path.join(results_dir, "mlp_pred_vs_true.png"),
        "MLP Regression: True vs Predicted"
    )

    print("\nDone. Results saved in hw1/results/")
    print(f"Linear Regression MSE: {lr_mse:.4f}")
    print(f"MLP Regression MSE: {mlp_mse:.4f}")


if __name__ == "__main__":
    main()