import numpy as np

def rbf_kernel(X1, X2, gamma=0.1):
    """
    RBF (Gaussian) カーネル: K(x, y) = exp(-γ ||x - y||²)
    """
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    dist_sq = X1_sq - 2 * np.dot(X1, X2.T) + X2_sq # X1とX2のユークリッド距離を計算
    return np.exp(-gamma * dist_sq)
