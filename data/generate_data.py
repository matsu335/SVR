import numpy as np
import os
import pandas as pd
from itertools import permutations
from team_ids import get_team_ids

# チームID辞書を取得
team_ids = get_team_ids()

# 出力先
output_dir = "match_results"
os.makedirs(output_dir, exist_ok=True)

# 勝率ファイル読み込み
winrate_df = pd.read_csv("winrates.csv")
winrate_dict = {
    (row['team1'], row['team2']): row['winrate']
    for _, row in winrate_df.iterrows()
}

# 全順序付きペアに対して処理
for (name1, id1), (name2, id2) in permutations(team_ids.items(), 2):
    file1 = f"regular_season_team_{id1}.csv"
    file2 = f"regular_season_team_{id2}.csv"

    if not (os.path.isfile(file1) and os.path.isfile(file2)):
        print(f"Skipping: {file1} or {file2} not found.")
        continue

    arr1 = np.loadtxt(file1, delimiter=",")
    arr2 = np.loadtxt(file2, delimiter=",")

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(arr1, arr2)
        ratio[~np.isfinite(ratio)] = np.nan

    # 勝率を取得
    winrate = winrate_dict.get((name1, name2), None)

    if winrate is None:
        print(f"Winrate missing for {name1} vs {name2}. Skipping.")
        continue

    # 結果を1つの配列にして保存（例：先頭にwinrateを加える）
    combined = np.concatenate(([winrate], ratio.flatten()))

    # 出力ファイル
    output_path = os.path.join(output_dir, f"{name1}_div_{name2}.csv")
    np.savetxt(output_path, combined.reshape(1, -1), delimiter=",", fmt="%.4f")