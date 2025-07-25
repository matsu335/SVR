import numpy as np
from svr import KernelSVR
from kernels import rbf_kernel
from data_loader import load_match_results_data
from sklearn.metrics import mean_squared_error

np.random.seed(42)

def main():
    X, y = load_match_results_data("data/match_results")

    # ランダムにインデックスをシャッフル
    total_samples = len(X)
    indices = np.random.permutation(total_samples)

    # 70個を学習データ、残り20個をテストデータに分割
    train_indices = indices[:70]
    test_indices = indices[70:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    model = KernelSVR(kernel=rbf_kernel, C=0.2154, epsilon=0.12, kernel_params={"gamma": 1.6681})
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred) # 平均二乗誤差MSEを計算。ライブラリを使用。
    print("MSE:", mse)
    print(f"y_test: {y_test}, y_pred: {y_pred}")

if __name__ == "__main__":
    main()
