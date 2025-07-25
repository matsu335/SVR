# kernel_svr.py
import numpy as np
from cvxopt import matrix, solvers

class KernelSVR:
    def __init__(self, kernel, C=1.0, epsilon=0.1, kernel_params={}):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.kernel_params = kernel_params

    def fit(self, X, t):
        self.X = X
        self.t = t
        n_samples = X.shape[0] # サンプル数
        K = self.kernel(X, X, **self.kernel_params) # カーネルを作成

        """
        ・solvers.qp(P, q, G, h, A, b)について
        以下の2次計画問題を解くライブラリ

        minimize 1/2*x.T*P*x +q.T*x
        subject to G*x <= h
                   A*x = b 

        今回のパラメータは以下
        x ← a
            a_hat
        P ←  K -K
            -K  K
        q ← ε-t
            ε+t
        G ←  I  0
             0  I
            -I  0
             0 -I
        h ← C
            C
            0
            0
        A ← [1, 1,..., 1, -1,..., -1]
        b ← 0
        """

        P = np.vstack([
            np.hstack([K, -K]),
            np.hstack([-K, K])
        ])
        P = matrix(P.astype(np.double))
        q = matrix((self.epsilon - t).tolist() + (self.epsilon + t).tolist())

        G_std = np.vstack([np.eye(2 * n_samples), -np.eye(2 * n_samples)])
        h_std = np.hstack([np.ones(2 * n_samples) * self.C, np.zeros(2 * n_samples)])
        G = matrix(G_std)
        h = matrix(h_std)

        A = matrix(np.hstack([np.ones(n_samples), -np.ones(n_samples)]), (1, 2 * n_samples))
        b = matrix(0.0)

        sol = solvers.qp(P, q, G, h, A, b) # 前述したパラメータの2次計画問題を解く
        a_vector = np.array(sol['x']).flatten() # 解となるxを取り出す
        self.a = a_vector[:n_samples] # aにあたる成分を抽出
        self.a_hat = a_vector[n_samples:] # a_hatにあたる成分を抽出

        self.support_indices = np.where((self.a - self.a_hat) != 0)[0] # サポートベクトルのインデックスを取得
        self.b = np.mean([
            t[i] - np.sum((self.a - self.a_hat) * K[i])
            for i in self.support_indices
        ]) # バイアス項bの計算

    def predict(self, X_test):
        K = self.kernel(X_test, self.X, **self.kernel_params)
        return np.dot((self.a - self.a_hat), K.T) + self.b
