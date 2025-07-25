import numpy as np
from svr import KernelSVR
from kernels import rbf_kernel
from data_loader import load_match_results_data
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X, y = load_match_results_data("data/match_results")

# ランダムにインデックスをシャッフル
total_samples = len(X)
indices = np.random.permutation(total_samples)

# 70個を学習データ、残り20個をテストデータに分割
train_indices = indices[:70]
test_indices = indices[70:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# 探索範囲
C_range = np.logspace(-2, 2, 10)
epsilon_range = np.linspace(0.01, 1.0, 10)
gamma_range = np.logspace(-2, 2, 10)     

# 最小MSEを初期化
best_mse = float('inf')
best_params = {}

# 探索ループ
for C in C_range:
    for epsilon in epsilon_range:
        for gamma in gamma_range:
            model = KernelSVR(kernel=rbf_kernel, C=C, epsilon=epsilon, kernel_params={"gamma": gamma})
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            if mse < best_mse:
                best_mse = mse
                best_params = {
                    "C": C,
                    "epsilon": epsilon,
                    "gamma": gamma
                }

# 結果表示
print("Best parameters:")
print(f"C = {best_params['C']:.4f}, epsilon = {best_params['epsilon']:.4f}, gamma = {best_params['gamma']:.4f}")
print(f"Lowest MSE: {best_mse:.4f}")