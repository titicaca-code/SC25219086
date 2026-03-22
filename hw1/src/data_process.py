import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    split_idx = int(len(df) * 0.8)

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    return (
        df,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        y_train_scaled,
        y_test_scaled,
        x_scaler,
        y_scaler,
    )