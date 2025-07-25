import numpy as np
import os
from svr import KernelSVR
from kernels import rbf_kernel
from data_loader import load_match_results_data

def main():
    X, y = load_match_results_data("data/match_results")

    X_train = X
    y_train = y

    model = KernelSVR(kernel=rbf_kernel, C=0.2154, epsilon=0.12, kernel_params={"gamma": 1.6681})
    # model = KernelSVR(kernel=rbf_kernel, C=0.2154, epsilon=0.12, kernel_params={"gamma": 0.05})
    model.fit(X_train, y_train)

    # 予測対象のファイル名（拡張子つき）
    target_files = ["jtekt_stings_div_tokyo_greatbears.csv", "tokyo_greatbears_div_jtekt_stings.csv", "wolfdogs_div_hiroshima_thunders.csv", "hiroshima_thunders_div_wolfdogs.csv",
                    "osaka_bluteon_div_jtekt_stings.csv", "jtekt_stings_div_osaka_bluteon.csv", "suntory_sunbirds_div_wolfdogs.csv", "wolfdogs_div_suntory_sunbirds.csv",
                    "suntory_sunbirds_div_jtekt_stings.csv", "jtekt_stings_div_suntory_sunbirds.csv"]  # ここに対象ファイル名を入れる

    X_test = []
    used_filenames = []

    for fname in target_files:
        path = os.path.join("data/match_results", fname)
        if os.path.isfile(path):
            arr = np.loadtxt(path, delimiter=",")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X_test.append(arr[0, 1:])  # ratio部分のみ
            used_filenames.append(fname)
        else:
            print(f"ファイルが見つかりません: {fname}")

    if not X_test:
        print("対象ファイルが見つかりませんでした。")
        return

    X_test = np.array(X_test)
    y_pred = model.predict(X_test)

    for fname, pred in zip(used_filenames, y_pred):
        print(f"{fname}: y_pred = {pred:.4f}")

if __name__ == "__main__":
    main()
