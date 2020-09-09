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

# # 対戦ゲームデータ分析甲子園
# - https://prob.space/competitions/game_winner/data/62

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

## test用
#train_df = train_df.head(1000)
#test_df = test_df.head(100)
#df_all = train_df.append(test_df).reset_index(drop=True)
# -

# # データ確認

def df_info(df):
    df.info()
    display(df.head().style.background_gradient(cmap="Pastel1"))
    display(df.describe().T.style.background_gradient(cmap="Pastel1"))

    # カラム名 / カラムごとのユニーク値数 / 最も出現頻度の高い値 / 最も出現頻度の高い値の出現回数 / 欠損損値の割合 / 最も多いカテゴリの割合 / dtypes を表示
    stats = []
    for col in df.columns:
        stats.append(
            (
                col,
                df[col].nunique(),
                df[col].value_counts().index[0],
                df[col].value_counts().values[0],
                df[col].isnull().sum() * 100 / df.shape[0],
                df[col].value_counts(normalize=True, dropna=False).values[0] * 100,
                df[col].dtype,
            )
        )
    stats_df = pd.DataFrame(
        stats,
        columns=[
            "Feature",
            "Unique values",
            "Most frequent item",
            "Freuquence of most frequent item",
            "Percentage of missing values",
            "Percentage of values in the biggest category",
            "Type",
        ],
    )
    display(stats_df.sort_values("Percentage of missing values", ascending=False).style.background_gradient(cmap="Pastel1"))


df_info(train_df)

df_info(test_df)

df_info(df_all)


# +
def check_dup_nan(df):
    # 重複データ確認
    _df = df.copy()  # 重複削除する場合はcopy不要
    print(f"shape before drop: {_df.shape}")
    _df.drop_duplicates(inplace=True)
    _df.reset_index(drop=True, inplace=True)
    print(f"shape after drop: {_df.shape}")

    # 欠損データ確認
    display(pd.DataFrame({"is_null": _df.isnull().sum()}))

_df = train_df.copy().drop(["id"], axis=1)
check_dup_nan(_df)
    
_df = test_df.copy().drop(["id"], axis=1)
check_dup_nan(_df)
    
_df = df_all.copy().drop(["id"], axis=1)
check_dup_nan(_df)
# -

# - idとラベル除いたら、train内だけで重複レコードあり
# - idとラベル除いたら、trainとtestでも重複レコードあり
# - id除いただけでも、重複レコードあり。消すべきレコード
#
# <br>
#
# - 欠損はrankが多いのはウデマエがランクつくレベルにきてないユーザだからか？ウデマエはC-が一番低くてXが一番高いみたい
#     - https://wiki.denfaminicogamer.jp/Splatoon2/%E3%82%A6%E3%83%87%E3%83%9E%E3%82%A8
#
# <br>
#     
# - levelがゲーム内の経験値（ランク）っぽい。ランクはやればやるほど上がり、ウデマエは試合に負けると下がるみたい
#     - https://www.nintendo.co.jp/switch/aab6a/battle/index.html
#
# <br>
#
# - 武器の欠損はメンバーが4人集まらなかったから？
#
# <br>
#
# - modeがバトルモードみたい。ナワバリバトルは4対4らしい
#     - https://wiki.denfaminicogamer.jp/Splatoon2/%E3%82%A6%E3%83%87%E3%83%9E%E3%82%A8
#
# <br>
#
#

# # 前処理

# +
# 重複レコード確認して消す
_df = train_df.drop(["id"], axis=1)
_df_dup = _df[_df.duplicated()]
display(_df_dup)

# 重複レコード削除
print(train_df.shape)
train_df = train_df[~train_df.drop(["id"], axis=1).duplicated()]
print(train_df.shape)
df_all = train_df.append(test_df).reset_index(drop=True)
# -

# 値がすべて同じ game-ver, lobbyは消しとく。idも一意なだけで多分使わないから消しとく
df_all = df_all.drop(["id", "game-ver", "lobby"], axis=1)
print(df_all.shape)


# +
# 時刻列ばらしておく
def time_cols(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col]) # dtype を datetime64 に変換
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['weekend'] = (df[date_col].dt.dayofweek.values >= 5).astype(int)
    df['hour'] = df[date_col].dt.hour
    return df


df_all = time_cols(df_all, 'period')
print(df_all.shape)
# -

display(df_all.head(3))

# # EDA

# ラベル不均衡でない
df_all["y"].value_counts().plot.bar()



# +
def compare_train_test_cate(col):
    """trainとtestでカテゴリ列の分布比較"""
    plt.figure(figsize=(18, 4))
    ax1 = train_df[col].value_counts().plot.bar(colormap='Paired', title=f"{col}")# , logy=True)
    test_df[col].value_counts().plot.bar(ax=ax1)#, logy=True)
    ax1.legend(["train", "test"]);
    plt.show()
    plt.clf()
    plt.close()
    
cate_cols = test_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
cate_cols.remove("period")
for col in cate_cols:
    print(col)
    compare_train_test_cate(col)


# +
def compare_train_test_num(col):#, is_log=True):
    """trainとtestで数値列の分布比較"""
    plt.figure(figsize=(18, 4))
    #if is_log:
    #    # 対数化
    #    train_df[col] = np.log1p(train_df[col])
    #    test_df[col] = np.log1p(test_df[col])
    ax1 = train_df[col].plot.hist(colormap='Paired', title=f"{col}", bins=100, logy=True)
    test_df[col].plot.hist(ax=ax1, bins=100, logy=True)
    ax1.legend(["train", "test"]);
    plt.show()
    plt.clf()
    plt.close()
    
num_cols = test_df.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"]).columns.to_list()
num_cols.remove("id")
for col in num_cols:
    print(col)
    compare_train_test_num(col)
# -

df_all[df_all["A1-level"] > 500]



# ランクちょっと違うユーザ同士もマッチングする
df_gachi = df_all[df_all["lobby-mode"] == "gachi"]
#display(df_gachi)
df_gachi[
    (df_gachi["A1-rank"] != df_gachi["A2-rank"])
    | (df_gachi["A1-rank"] != df_gachi["A3-rank"])
    | (df_gachi["A1-rank"] != df_gachi["A4-rank"])
    | (df_gachi["A1-rank"] != df_gachi["B1-rank"])
    | (df_gachi["A1-rank"] != df_gachi["B2-rank"])
    | (df_gachi["A1-rank"] != df_gachi["B3-rank"])
    | (df_gachi["A1-rank"] != df_gachi["B4-rank"])
    | (df_gachi["A2-rank"] != df_gachi["A3-rank"])
    | (df_gachi["A2-rank"] != df_gachi["A4-rank"])
    | (df_gachi["A2-rank"] != df_gachi["B1-rank"])
    | (df_gachi["A2-rank"] != df_gachi["B2-rank"])
    | (df_gachi["A2-rank"] != df_gachi["B3-rank"])
    | (df_gachi["A2-rank"] != df_gachi["B4-rank"])
    | (df_gachi["A3-rank"] != df_gachi["A4-rank"])
    | (df_gachi["A3-rank"] != df_gachi["B1-rank"])
    | (df_gachi["A3-rank"] != df_gachi["B2-rank"])
    | (df_gachi["A3-rank"] != df_gachi["B3-rank"])
    | (df_gachi["A3-rank"] != df_gachi["B4-rank"])
    | (df_gachi["A4-rank"] != df_gachi["B1-rank"])
    | (df_gachi["A4-rank"] != df_gachi["B2-rank"])
    | (df_gachi["A4-rank"] != df_gachi["B3-rank"])
    | (df_gachi["A4-rank"] != df_gachi["B4-rank"])
    | (df_gachi["B1-rank"] != df_gachi["B2-rank"])
    | (df_gachi["B1-rank"] != df_gachi["B3-rank"])
    | (df_gachi["B1-rank"] != df_gachi["B4-rank"])
    | (df_gachi["B2-rank"] != df_gachi["B3-rank"])
    | (df_gachi["B2-rank"] != df_gachi["B4-rank"])
    | (df_gachi["B3-rank"] != df_gachi["B4-rank"])
]





def check_countplot_col_y(df, cate_col):
    """ラベルとカテゴリ列のカウント数plot"""
    target = "y"
    for i, _df in df.groupby(cate_col):
        m = _df.iloc[0][cate_col]
        _df = _df.rename(columns={target: f"{m}_{target}"})
        sns.countplot(_df[f"{m}_{target}"])
        plt.show()
        plt.clf()
        plt.close()


# バトルモードで勝敗の差若干ある
check_countplot_col_y(df_all, "mode")

# lobby-modeで勝敗の差若干ある。
# セグメンテーションしようかな
check_countplot_col_y(df_all, "lobby-mode")

# stageで勝敗の差若干ある
check_countplot_col_y(df_all, "stage")

# yearで勝敗の差若干ある。期間は2019-2020年
check_countplot_col_y(df_all, "year")

# monthで勝敗の差若干ある。期間は2019/10-2020/01
check_countplot_col_y(df_all, "month")

# dayで勝敗の差若干ある。16dayだけy=0.0のほうが多い
check_countplot_col_y(df_all, "day")

# dayofyearで勝敗の差若干ある。6dayofyearだけ異様にデータ数少ない。メンテとかはいったんかな？
check_countplot_col_y(df_all, "dayofyear")

# 6dayofyearのレコード。hour0の時間しかないので多分メンテ日だな
df_all[df_all["dayofyear"] == 6] 

# dayofweek。曜日による勝敗の比の違いはない。週末は利用者増える
check_countplot_col_y(df_all, "dayofweek")

# weekend。週末による勝敗の比の違いはない。週末は利用者多いな
check_countplot_col_y(df_all, "weekend")

# hourで勝敗の差若干ある。2時間おきに集計してるみたい。14時の時間帯が利用ピーク
check_countplot_col_y(df_all, "hour")

# - この時間だから勝ちやすいとかあるのかな。。。強い人がA,Bのどっちかのチーム偏る時間や日があるとか？



# +
# rank="x"は特別なウデマエみたい
# https://pcfreebook.com/article/spla2-udemaex.html
# ウデマエXとは、スプラトゥーンのガチマッチにおける階級の最高位（最高ランク）です。
# S+9のときにウデマエゲージをMAXにすると(S+10以上にする)、ウデマエXへと昇格します。
# ウデマエXの状態では、ガチマッチはウデマエX同士でしかマッチングしなくなり、Xパワーと呼ばれるものを賭けて戦ういわゆる「レート戦」をすることになります。

# 実際、rank=xのレコードはほかのrankとマッチングしていない
df_all_x = df_all[df_all["A1-rank"] == "x"]
df_info(df_all_x)
check_countplot_col_y(df_all_x, "lobby-mode")
# -

# rank=bについて
# 同じレベルのrank者同士がマッチングするようになってるみたい
df_all_b = df_all[
    (df_all["A1-rank"] == "b")
    | (df_all["A1-rank"] == "b-")
    | (df_all["A1-rank"] == "b+")
]
df_info(df_all_b)
check_countplot_col_y(df_all_b, "lobby-mode")

# rank=cについて
# 同じレベルのrank者同士がマッチングするようになってるみたい
df_all_c = df_all[
    (df_all["A1-rank"] == "c")
    | (df_all["A1-rank"] == "c-")
    | (df_all["A1-rank"] == "c+")
]
display(df_all_c)
print(df_all_c.shape)
check_countplot_col_y(df_all_c, "lobby-mode")



# A1-rankが欠損はレギューラーマッチのみ。レギューラーマッチはランク関係ないから？
df_all_n = df_all[df_all["A1-rank"].isnull()]
display(df_all_n)
check_countplot_col_y(df_all_n, "lobby-mode")

# A4-rankが欠損はガチマッチとレギューラーマッチ両方
df_all_n = df_all[df_all["A4-rank"].isnull()]
display(df_all_n)
check_countplot_col_y(df_all_n, "lobby-mode")

# A4-rankが欠損のガチマッチは人数3人とかのときだけ
df_all_n[df_all_n["lobby-mode"] == "gachi"]

# B4-rankが欠損はガチマッチとレギューラーマッチ両方
df_all_n = df_all[df_all["B4-rank"].isnull()]
display(df_all_n)
check_countplot_col_y(df_all_n, "lobby-mode")

# B4-rankが欠損のガチマッチは人数3人とかのときだけ
df_all_n[df_all_n["lobby-mode"] == "gachi"]



# +
# A1-levelは外れ値ありそう
def check_out(df):
    # 外れ値確認
    num_cols = df.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"]).columns.to_list()
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=df[num_cols])
    plt.show()
    plt.clf()
    plt.close()
    
# level列だけで確認
l_in = [s for s in df_all.columns.to_list() if 'level' in s]
check_out(df_all[l_in])


# + hide_input=false
# A2-level, A3-level, A4-level, B1-level, B2-level, B3-level, B4-level は対数化した方が良いかも
def check_num_hist(df):
    # 数値列のヒストグラム。歪み（Skewness）大きい場合は対数変換した方が良い
    num_cols = df.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"]).columns.to_list()
    for col in num_cols:
        try:
            _ser = df[col].dropna()
            plt.figure(figsize=(16, 4))
            sns.distplot(_ser, label=f"Skewness: {round(_ser.skew(), 2)}")
            plt.legend()
            plt.show()
            plt.clf()
            plt.close()
        except Exception as e:
            print(e)
            print(f"ERROR: {col}")

l_in = [s for s in df_all.columns.to_list() if 'level' in s]
check_num_hist(df_all[l_in])
# +
# 歪度あまり改善しなので対数化しても意味なさそう
def log1p(df, num_cols):
    for col in num_cols:
        df[f"{col}_log1p"] = np.log(df[col] + 1)
    return df

log1p_cols = ["A2-level", "A3-level", "A4-level", "B1-level", "B2-level", "B3-level", "B4-level"]
df_all = log1p(df_all, log1p_cols)

new_log1p_cols = [f"{c}_log1p" for c in log1p_cols]
check_num_hist(df_all[new_log1p_cols])


# +
def check_cate_count(df):
    # カテゴリ列のレコード数
    cate_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    for col in cate_cols:
        plt.figure(figsize=(18, 5))
        sns.countplot(x=col, data=df, color='salmon')
        plt.show()
        plt.clf()
        plt.close()
        
check_cate_count(df_all)

# +
# trainとtestの期間はごちゃ混ぜだった
# cvは時系列のやり方で分けなくて良さそう
train_df['period'] = pd.to_datetime(train_df['period'], format='%Y-%m-%d %H:%M:%S')
test_df['period'] = pd.to_datetime(test_df['period'], format='%Y-%m-%d %H:%M:%S')
df_all['period'] = pd.to_datetime(df_all['period'], format='%Y-%m-%d %H:%M:%S')

train_df = train_df.set_index('period', drop=False)
test_df = test_df.set_index('period', drop=False)
df_all = df_all.set_index('period', drop=False)

display(df_all.head(1))

# 一意のid降りなおす
df_all["id"] = list(range(df_all.shape[0])) 

train_df["id"].plot(title="train_id", figsize=(16, 6))
plt.show()

test_df["id"].plot(title="test_df_id", figsize=(16, 6))
plt.show()

df_all["id"].plot(title="df_all_id", figsize=(16, 6))
plt.show()
# -

# - 

# カテゴリ列 vs. カテゴリ列。これは１列ごとに見てかないと理解できない
# countの場合は valuesはどの列選んでも同じ。columnsとindexが一意なレコードの数数える
str_cols = df_all.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
for col in str_cols:
    _df = df_all.pivot_table(values="id", index=col, columns="y", aggfunc="count",)
    _df.plot.bar(figsize=(18, 6))
    plt.show()
    plt.clf()
    plt.close()

# - rankによる勝ち負けの差は大きくはなさそう
# - 武器によって勝ち負けの差大きそう
# - mode違いで勝ち負けの差は少しありそう
# - ステージ違いで勝ち負けの差は大きくはなさそう

# +
# Aチームの武器の分布
v_max = df_all["A1-weapon"].value_counts().max()

for (i, _df_A1), (_, _df_A2), (_, _df_A3), (_, _df_A4) in zip(
    df_all.groupby("A1-weapon"),
    df_all.groupby("A2-weapon"),
    df_all.groupby("A3-weapon"),
    df_all.groupby("A4-weapon"),
):
    fig, ax = plt.subplots(1, 4, figsize=(20, 3))
    ti = _df_A1["A1-weapon"].iloc[0]
    _df_A1["y"].value_counts().plot.bar(title=f"A1_{ti}", ylim=[0, v_max // 2], ax=ax[0])
    _df_A2["y"].value_counts().plot.bar(title=f"A2_{ti}", ylim=[0, v_max // 2], ax=ax[1])
    _df_A3["y"].value_counts().plot.bar(title=f"A3_{ti}", ylim=[0, v_max // 2], ax=ax[2])
    _df_A4["y"].value_counts().plot.bar(title=f"A4_{ti}", ylim=[0, v_max // 2], ax=ax[3])
    plt.show()
    plt.clf()
    plt.close()
# +
# Bチームの武器の分布
v_max = df_all["B1-weapon"].value_counts().max()

for (i, _df_B1), (_, _df_B2), (_, _df_B3), (_, _df_B4) in zip(
    df_all.groupby("B1-weapon"),
    df_all.groupby("B2-weapon"),
    df_all.groupby("B3-weapon"),
    df_all.groupby("B4-weapon"),
):
    fig, ax = plt.subplots(1, 4, figsize=(20, 3))
    ti = _df_B1["B1-weapon"].iloc[0]
    _df_B1["y"].value_counts().plot.bar(title=f"B1_{ti}", ylim=[0, v_max // 2], ax=ax[0])
    _df_B2["y"].value_counts().plot.bar(title=f"B2_{ti}", ylim=[0, v_max // 2], ax=ax[1])
    _df_B3["y"].value_counts().plot.bar(title=f"B3_{ti}", ylim=[0, v_max // 2], ax=ax[2])
    _df_B4["y"].value_counts().plot.bar(title=f"B4_{ti}", ylim=[0, v_max // 2], ax=ax[3])
    plt.show()
    plt.clf()
    plt.close()
# -

# - よく使われる武器とそうでない武器の差が激しい
# - どの武器もラベルに不均衡はなさそう

# 特徴量同士の相関ヒートマップ
# 元々の列で相関明らかに高い列はない
fig, ax = plt.subplots(figsize=(25, 14))
sns.heatmap(df_all.corr(), square=True, vmax=1, vmin=-1, center=0, annot=True)
plt.show()

# +
# カテゴリ列 vs. 複数の数値列
cate_col = "y"
num_columns = df_all.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"]).columns.to_list()

for i, n_col in enumerate(num_columns):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(n_col)
    sns.boxplot(x=cate_col, y=n_col, data=df_all, ax=ax[0])
    sns.violinplot(x=cate_col, y=n_col, data=df_all, ax=ax[1])
    plt.tight_layout(rect=[0,0,1,0.96])  # タイトル重ならないようにする
    plt.show()
    plt.clf()
    plt.close()


# + code_folding=[0, 18]
def grouping(df, cols, agg_dict, prefix=""):
    """特定のカラムについてgroup化された特徴量の作成を行う
    Args:
        df (pd.DataFrame): 特徴量作成のもととなるdataframe
        cols (str or list): group by処理のkeyとなるカラム (listで複数指定可能)
        agg_dict (dict): 特徴量作成を行いたいカラム/集計方法を指定するdictionary
        prefix (str): 集約後のカラムに付与するprefix name

    Returns:
        df (pd.DataFrame): 特定のカラムについてgroup化された特徴量群
    """
    group_df = df.groupby(cols).agg(agg_dict)
    group_df.columns = [prefix + c[0] + "_" + c[1] for c in list(group_df.columns)]
    group_df.reset_index(inplace=True)

    return group_df


def plot_target_stats(df, col, agg_dict, plot_config, figsize=(15, 8)):
    """指定したカラムとtargetの関係を可視化する
    Args:
        df (pd.DataFrame): 可視化したい特徴量作成のもととなるdataframe
        col (str): group by処理のkeyとなるカラム
        agg_dict (dict): 特徴量作成を行いたいカラム/集計方法を指定するdictionary

    """

    plt_data = grouping(df, col, agg_dict, prefix="")

    target_col = list(agg_dict.keys())[0]

    # 2軸グラフの作成
    fig, ax1 = plt.subplots(figsize=figsize)

    ax2 = ax1.twinx()

    ax1.bar(
        plt_data[col],
        plt_data[f"{target_col}_count"],
        label=f"{target_col}_count",
        color="skyblue",
        **plot_config["bar"],
    )
    ax2.plot(
        plt_data[col],
        plt_data[f"{target_col}_mean"],
        label=f"{target_col}_mean",
        color="red",
        marker=".",
        markersize=10,
    )

    h1, label1 = ax1.get_legend_handles_labels()
    h2, label2 = ax2.get_legend_handles_labels()

    ax1.legend(h1 + h2, label1 + label2, loc=2, borderaxespad=0.0)
    ax1.set_xticks(plt_data[col])
    ax1.set_xticklabels(plt_data[col], rotation=-90, fontsize=10)

    ax1.set_title(
        f"Relationship between {col}, {target_col}_count, and {target_col}_mean",
        fontsize=14,
    )
    ax1.set_xlabel(f"{col}")
    ax1.tick_params(labelsize=12)

    ax1.set_ylabel(f"{target_col}_count")
    ax2.set_ylabel(f"{target_col}_mean")

    ax1.set_ylim([0, plt_data[f"{target_col}_count"].max() * 1.2])
    ax2.set_ylim([0, plt_data[f"{target_col}_mean"].max() * 1.1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # タイトル重ならないようにする
    plt.show()
    plt.clf()
    plt.close()

# 各カラムごとのレコード数と数値型の目的変数の平均値を可視化
target_col = "y"
for col in df_all.columns.to_list():
    # 種別1000未満の列のみ可視化
    n_uni = df_all[col].value_counts().shape[0]
    if n_uni < 1000:
        agg_dict = {target_col: ["count", "mean"]}
        plot_config = {"bar": {"width": 0.8}}
        plot_target_stats(df_all, col, agg_dict, plot_config)


# -

# - レベル高いレコードは少ないので平均の値が極端
# - A1,A2,A3,A4,B1,B2,B3,B4それぞれレベルの分布が違う。A1はレベル高いレコード多めだが、ほかのはレベル高いのは少ない

# +
def c_pairplot(df):
    # カスタムペアプロット
    import sys
    sys.path.append(r'C:\Users\81908\Git\seaborn_analyzer')

    from custom_pair_plot import CustomPairPlot
    # 行数列数多いと処理終わらないので注意
    num_columns = df.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"]).columns.to_list()
    CustomPairPlot().pairanalyzer(df[num_columns])  # dfでも実行できる
    plt.show()
    
c_pairplot(df_all)
# -
# # 追加データ見てみる

df_add = pd.read_csv(f"{ORIG}/statink-weapon2.csv")
df_info(df_add)


# +
def merage_df_add(df_all, df_add, pre):
    """武器情報の追加ファイルを結合"""
    df_add = df_add[
        ["key", "category1", "category2", "mainweapon", "subweapon", "special", "splatnet"]
    ]
    df_add.columns = [pre + col for col in df_add.columns]
    return pd.merge(df_all, df_add, how="left", left_on=f"{pre}weapon", right_on=f"{pre}key").drop(columns=[f"{pre}key"])

    
df_add = pd.read_csv(f"{ORIG}/statink-weapon2.csv")
for pre in ["A1-", "A2-", "A3-", "A4-", "B1-", "B2-", "B3-", "B4-"]:
    df_all = merage_df_add(df_all, df_add, pre)
df_info(df_all)
# -







# +
from IPython.display import HTML

HTML("""
<button id="code-show-switch-btn">スクリプトを非表示にする</button>

<script>
var code_show = true;

function switch_display_setting() {
    var switch_btn = $("#code-show-switch-btn");
    if (code_show) {
        $("div.input").hide();
        code_show = false;
        switch_btn.text("スクリプトを表示する");
    }else {
        $("div.input").show();
        code_show = true;
        switch_btn.text("スクリプトを非表示にする");
    }
}

$("#code-show-switch-btn").click(switch_display_setting);
</script>
""")



