import numpy as np
import matplotlib.pyplot as plt
from svr import KernelSVR
from kernels import rbf_kernel
from data_loader import load_match_results_data
from sklearn.metrics import mean_squared_error

# データの準備
np.random.seed(42)
X, y = load_match_results_data("data/match_results")

# ランダムにインデックスをシャッフル
total_samples = len(X)
indices = np.random.permutation(total_samples)

train_indices = indices[:70]
test_indices = indices[70:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# 最適パラメータ
best_params = {"C": 0.2154, "epsilon": 0.12, "gamma": 1.6681}

# 探索範囲
C_range = np.logspace(-2, 2, 50)
epsilon_range = np.linspace(0.01, 1.0, 50)
gamma_range = np.logspace(-2, 2, 50)

# MSEを格納する関数
def evaluate_param(param_name, values):
    mses = []
    for val in values:
        params = best_params.copy()
        params[param_name] = val
        model = KernelSVR(kernel=rbf_kernel, C=params["C"], epsilon=params["epsilon"], kernel_params={"gamma": params["gamma"]})
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mses.append(mse)
    return mses

# それぞれの変数についてMSEを算出
mse_C = evaluate_param("C", C_range)
mse_epsilon = evaluate_param("epsilon", epsilon_range)
mse_gamma = evaluate_param("gamma", gamma_range)

# グラフの描画
plt.figure(figsize=(18, 5))

# C のグラフ
plt.subplot(1, 3, 1)
plt.plot(C_range, mse_C, marker='o')
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("MSE")
plt.title("MSE vs C (epsilon, gamma fixed)")

# epsilon のグラフ
plt.subplot(1, 3, 2)
plt.plot(epsilon_range, mse_epsilon, marker='o')
plt.xlabel("epsilon")
plt.ylabel("MSE")
plt.title("MSE vs epsilon (C, gamma fixed)")

# gamma のグラフ
plt.subplot(1, 3, 3)
plt.plot(gamma_range, mse_gamma, marker='o')
plt.xscale("log")
plt.xlabel("gamma")
plt.ylabel("MSE")
plt.title("MSE vs gamma (C, epsilon fixed)")

plt.tight_layout()
plt.show()
