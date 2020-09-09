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

# + hide_input=false
# !pwd
import sys
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
sys.executable

# +
import os
import gc
import sys
import joblib
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import *
from sklearn.metrics import *
import lightgbm as lgb
from lightgbm import *

sys.path.append(r"C:\Users\81908\Git\xfeat")
import xfeat
from xfeat import *
from xfeat.selector import *
from xfeat.utils import compress_df

sns.set()
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)

# +
OUT_DATA = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\work"
os.makedirs(OUT_DATA, exist_ok=True)

ORIG = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\orig"
train_df = pd.read_csv(f"{ORIG}/train_data.csv")
test_df = pd.read_csv(f"{ORIG}/test_data.csv")
df_all = train_df.append(test_df).reset_index(drop=True)

# +
# 重複レコード確認して消す
_df = train_df.drop(["id"], axis=1)
_df_dup = _df[_df.duplicated()]
display(_df_dup)

# 重複レコード削除
print(train_df.shape)
train_df = train_df[~train_df.drop(["id"], axis=1).duplicated()].reset_index(drop=True)
print(train_df.shape)
df_all = train_df.append(test_df).reset_index(drop=True)
# -

_DIR = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\model\train"
df_pred = pd.read_csv(f"{_DIR}/train_probas.tsv", sep=",")
display(df_pred)

# +
df_pred["cv_mean"] = df_pred.apply(lambda x: np.mean(x), axis=1)
_df = pd.concat([train_df, df_pred], axis=1)
_df["diff"] = _df["y"] - _df["cv_mean"]
_df["diff"] = np.abs(_df["diff"])
_df = _df.sort_values("diff", ascending=True)
#_df = _df.reset_index()
_df = _df.tail(20)
#_df["index"] = _df["index"].astype(str)
display(_df)
#_df["diff"].head(10)#.T.plot()

#plt.figure(figsize=(8, 10))
#sns.barplot(y="index", x="diff", data=_df)
#sns.barplot(x="index", y="diff", data=_df)
#plt.show()

_df["diff"].plot.barh(figsize=(8, 10))
#df_pred[["fold_1", "fold_2"]].plot.scatter(x="fold_1", y="fold_2", figsize=(10, 10))
#df_pred["fold_1"].plot(figsize=(10, 10))
# -

import sys
sys.path.append(r'C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\code')
from util import Util

Util.check_y_diff("tmp", 
                  r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\model\code_train\code_eng1_1_csv\train_best_param\train_probas.tsv")


# + code_folding=[3]
class Submit():

    @staticmethod
    def nelder_mead_th(true_y, pred_y):
        """ネルダーミードでf1スコアから2値分類のbestな閾値見つける"""
        from scipy.optimize import minimize

        def f1_opt(x):
            return -f1_score(true_y, pred_y >= x)

        result = minimize(f1_opt, x0=np.array([0.5]), method="Nelder-Mead")
        best_threshold = result["x"].item()
        return best_threshold


    @staticmethod
    def pred_nelder_mead(pred_dir, out_dir):
        """nelder_meadで決めた閾値でcvの平均値を2値化する"""
        # nelder_meadの学習データ
        df_pred = pd.read_csv(f"{pred_dir}/train_probas.tsv", sep=",")
        train_y = train_df["y"].values
        # cvの平均値
        train_pred_prob = df_pred.apply(lambda x: np.mean(x), axis=1).values

        # 閾値を0.5としたとき
        init_threshold = 0.5
        init_score = f1_score(train_y, train_pred_prob >= init_threshold)
        print("init_threshold, init_score:", init_threshold, init_score)

        # nelder_meadの閾値
        best_threshold = nelder_mead_th(train_y, train_pred_prob)
        best_score = f1_score(train_y, train_pred_prob >= best_threshold)
        print("best_threshold, best_score:", best_threshold, best_score)

        # nelder_meadの閾値で2値化
        df_pred = pd.read_csv(f"{pred_dir}/test_probas.tsv", sep=",")
        test_pred_prob = df_pred.apply(lambda x: np.mean(x), axis=1).values
        test_pred_y = [int(x > best_threshold) for x in test_pred_prob]

        # ファイル出力
        df_test_pred = pd.DataFrame({"y": test_pred_y}).reset_index()
        df_test_pred.columns = ["id", "y"]
        df_test_pred.to_csv(f"{out_dir}/submission_nelder_mead.csv", index=False)
        display(df_test_pred)
    

_DIR = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\model\train"
Submit().pred_nelder_mead(_DIR, _DIR)
# -

_DIR = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\model\train"
df_pred = pd.read_csv(f"{_DIR}/test_probas.tsv", sep=",")
pred_prob = df_pred.apply(lambda x: np.mean(x), axis=1).values
[int(x > 0.49111328125) for x in pred_prob]





# + code_folding=[]
def nelder_mead_func(trues, preds):
    """
    ネルダーミードで任意の関数のbestな重み見つける
    - func_opt関数の中身変更すること
    - 複数モデルの結果ブレンドするのに使える
    """
    from sklearn.metrics import f1_score
    from scipy.optimize import minimize
    from sklearn.metrics import mean_squared_error

    def func_opt(x):
        # y = a*x1 + b*x2 * c*x3 みたいな式のa,b,cのbest重み最適化
        blend_preds = 0
        for x_i, p_i in zip(x, preds):
            blend_preds += p_i * x_i
        print("p", len(blend_preds))
        print("t", len(trues))
        # 正解との平均2乗誤差返す
        return mean_squared_error(trues, blend_preds)

    result = minimize(func_opt, x0=np.array([1.0 for i in range(preds.shape[1])]), method="Nelder-Mead")
    # print(result)
    best_thresholds = result["x"]
    return best_thresholds

if __name__ == '__main__':
    train_y = train_df["y"].values
    #print(train_y)
    train_pred_probs = df_pred.values
    #print(train_pred_probs)
    
    # y = a*x1 + b*x2 * c*x3 の式のa,b,cのbest重み最適化
    best_thresholds = nelder_mead_func(train_y, train_pred_probs)    
    print("best_thresholds:", best_thresholds)
# -


