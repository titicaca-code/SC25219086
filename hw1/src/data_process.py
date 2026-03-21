import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)

    # 默认最后一列是目标值
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    split_idx = int(len(df) * 0.8)

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return df, X_train_scaled, X_test_scaled, y_train, y_test, scaler