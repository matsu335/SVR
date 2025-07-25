import os
import numpy as np

def load_match_results_data(data_dir):
    X = []
    y = []

    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            arr = np.loadtxt(os.path.join(data_dir, fname), delimiter=",")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            y.append(arr[0, 0])      # winrate
            X.append(arr[0, 1:])     # ratio

    return np.array(X), np.array(y)
