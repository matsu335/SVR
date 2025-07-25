import requests
from bs4 import BeautifulSoup
import numpy as np

def fetch_regular_season_array(team_id: int) -> np.ndarray:
    """
    指定されたチームIDのチーム詳細ページから
    『レギュラーシーズン』行の数値データをNumPy配列で返す。
    """
    url = f"https://www.svleague.jp/ja/sv_men/recordn/team_detail/{team_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.select_one("div.block_record_form_result table")
    if not table:
        raise ValueError("成績テーブルが見つかりませんでした。")

    data = []
    for row in table.find_all("tr", class_="t2"):
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if cols[0] == "レギュラーシーズン":
            # 1列目（大会名）を除いて数値列のみ抽出
            values = [float(x) for x in cols[1:]]
            data.append(values)

    if not data:
        raise ValueError("レギュラーシーズンの行が見つかりませんでした。")

    # 通常は1行なので1次元配列を返す
    arr = np.array(data[0], dtype=float)
    indices_to_remove = np.arange(len(arr) - 5, len(arr) - 2)
    arr_new = np.delete(arr, indices_to_remove)
    return arr_new

"""
arr_newのパラメータ
["試合数", "セット数", "アタック打数", "アタック得点", "アタック失点", "アタック決定率",
"バックアタック打数", "バックアタック得点", "バックアタック失点", "バックアタック決定率",
"アタック決定本数(セット平均)", "ブロック得点", "ブロックセット平均",
"サーブ打数", "サーブエース", "サーブ効果", "サーブ効果率",
"サーブレシーブ受数", "サーブレシーブ成功・優", "サーブレシーブ成功・良", "サーブレシーブ成功率",
"得点合計", "サーブ失点"]
"""

# if __name__ == "__main__":
#     team_id = 479  # 例：チームIDを指定
#     arr = fetch_regular_season_array(team_id)
#     print("レギュラーシーズンのデータ (NumPy 配列):")
#     print(arr)
