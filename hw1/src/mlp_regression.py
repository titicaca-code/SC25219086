import copy
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error


class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


def run_mlp_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_scaled,
    y_scaler,
    epochs=1000,
    lr=0.001
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从训练集里再切一部分做验证集
    val_ratio = 0.1
    split_idx = int(len(X_train) * (1 - val_ratio))

    X_tr = X_train[:split_idx]
    X_val = X_train[split_idx:]
    y_tr = y_train_scaled[:split_idx]
    y_val = y_train_scaled[split_idx:]

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = MLPRegressor(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_tr_tensor)
        train_loss = criterion(train_pred, y_tr_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss.item():.6f}, "
                f"Val Loss: {val_loss.item():.6f}"
            )

    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    mse = mean_squared_error(y_test, y_pred)

    return model, y_pred, mse