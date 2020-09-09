# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# !pwd
import sys
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
sys.executable

# # auc=0.502なので学習データとテストデータの分布が同じ

# + code_folding=[8]
import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def adversarial_validation(X_train, X_test):
    """
    adversarial_validation:学習データとテストデータの分布が同じか判断する手法
    学習データとテストデータを結合してテストデータか否かを目的変数とする二値分類
    同じ分布なら見分けられないのでauc=0.5に近くなる。こうなるのが理想
    0.5上回るなら違う分布
    """
    # 学習データのラベル=1, テストデータのラベル=0とする
    y_train = np.array([1 for i in range(X_train.shape[0])])
    y_test = np.array([0 for i in range(X_test.shape[0])])

    # 学習データとテストデータ結合
    y = np.concatenate([y_train, y_test])
    X = pd.concat([X_train, X_test])

    # ラベル付け替えたデータでtrain/testに分ける
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # lightGBM 用のデータに変形
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_test, y_test)

    # 二値分類モデル作成
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "n_jobs": -1,
        "seed": 236,
    }
    model_lgb = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        early_stopping_rounds=50,
        valid_sets=[lgb_train, lgb_val],
        verbose_eval=-1,
    )

    # AUC計算
    pred = model_lgb.predict(X_test)
    score = roc_auc_score(y_test, pred)
    print(f"AUC: {round(score, 3)}")

    
if __name__ == "__main__":   
    DATA_DIR = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\feature_eng"
    df = pd.read_csv(f"{DATA_DIR}/preprocess.csv", index_col=0)
    
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns.to_list():
        df[col], uni = pd.factorize(df[col])

    y_col = "y"
    drop_cols = [y_col]  # 取り除きたいカラムのリスト
    cols = [c for c in df.columns if c not in drop_cols]
    df_x = df[cols]
    X_train, X_test, y_train, y_test = train_test_split(
        df_x, df[y_col], random_state=42
    )
    
    adversarial_validation(X_train, X_test)
