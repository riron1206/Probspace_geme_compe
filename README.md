# Probspace_geme_compe

## pycaret2のモデルで43位だった
- https://prob.space/competitions/game_winner/submissions

<br>

- ランダム性が強いデータなので、たまたましてたcatboost,lightGBM,gcbのスタッキングによりシェイクダウンしなかったと思われる（pycaret-2-classification_code_eng2_csv.ipynb のstacker_submission.csv）

<br>

- 武器ごとの勝率の特徴量など入れたlightGBMのシングルモデルでも同じぐらいのPrivateスコアも出せてた（eng2.csv使ったもの。run_train_lgb.sh のcode_eng2_csv_best_params_eng2_csv_tuining()のはず）

<br>

- psude labelとかも試したがうまくいかなかった。他に試した手法
    - https://trello.com/b/ogXQKyvT/%E3%82%B9%E3%83%97%E3%83%A9%E3%82%B3%E3%83%B3%E3%83%9A%E3%81%AE%E3%82%A2%E3%82%A4%E3%83%87%E3%82%A2

## 上位手法メモ
- 11位
    - ドメイン知識関係ない特徴量として、AチームとBチームのデータを入れ替えることで，学習データを倍にした
    - https://prob.space/topics/takaito-Post28d53332b6f7a13654be
- 3位
    - ドメイン知識にもとづいて、2時間の中で同じ武器を何回使ったか、何種類の武器を使ったかCount Encoding
    - https://prob.space/topics/TASSAN-Postfea0f0d8cea0d9819c1c
