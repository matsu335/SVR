・ディレクトリ構造
SVR/
├── data/                    ← データ関係のファイルを入れるフォルダ
│   ├── match_results/       ← リザルトをcsvファイルとして格納するフォルダ。generate_data.pyにより出力される
│   ├── fetch_data.py        ← SVリーグの公式サイトからデータを持ってくる関数
│   ├── generate_data.py     ← 元データから学習用のデータを作り、match_resultsフォルダ内に格納
│   ├── ~.csv                ← save_data.pyによって作られるcsvファイル。各チームのリザルトを格納
│   ├── save_data.py         ← fetch_data.pyを使い持った来たデータをcsvファイルとして保存する
│   ├── team_ids.py          ← team_idを管理する辞書を取り出す関数
│   └── winrates.csv         ← 勝率を保存しておくファイル。手動で記入した
├── results/
│   └── MSE.png              ← 各変数ごとのMSEの変化をマッピングした図
├── src/
│   ├── data_loader.py       ← データ読み込みの関数
│   ├── graphing.py          ← 各ハイパラメータごとのMSEの変化をグラフ化する
│   ├── kernels.py           ← 用いるカーネルをまとめたファイル。今回はrbfカーネルのみ
│   ├── main.py              ← ある条件下での訓練・テストをし、mseを評価する
│   ├── predict_final.py     ← 本実験ではレギュラーラウンドのデータを用いたが、そこで得たモデルを用いてファイナルランドの結果を予測
│   ├── search_parameters.py ← mseを最小にするハイパラメータを探索する
│   └── svr.py               ← SVRモデルを実装したクラス
├── requirements.txt         ← 実行に必要なライブラリ等をまとめたファイル
└── README.md                ← このファイル。説明等を記入

・実行方法
1. requirements.txt内の必要ライブラリをインストール
2. 「python main.py」を実行し、mseを確認できる（main.py以外にもgraphing.pyやpredict_final.py、search_parameters.pyは実行可能）