import numpy as np
from fetch_data import fetch_regular_season_array
from team_ids import get_team_ids

# チーム名とIDの対応辞書を取得
team_ids = get_team_ids()

# 各チームについてデータを取得してCSVに保存
for team_name, team_id in team_ids.items():
    arr = fetch_regular_season_array(team_id)
    np.savetxt(
        f"data/regular_season_team_{team_id}.csv",
        arr.reshape(1, -1),
        delimiter=",",
        fmt="%.2f"
    )