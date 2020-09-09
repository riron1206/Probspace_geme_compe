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

OUT_MODEL = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\model\first"
os.makedirs(OUT_MODEL, exist_ok=True)

ORIG = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\orig"
train_df = pd.read_csv(f"{ORIG}/train_data.csv")
test_df = pd.read_csv(f"{ORIG}/test_data.csv")
df_all = train_df.append(test_df).reset_index(drop=True)


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

_df = train_df.copy().drop(["id", "y"], axis=1)
check_dup_nan(_df)
    
_df = test_df.copy().drop(["id"], axis=1)
check_dup_nan(_df)
    
_df = df_all.copy().drop(["id", "y"], axis=1)
check_dup_nan(_df)


# -

# - idとラベル除いたら、train内だけで重複レコードあり
# - idとラベル除いたら、trainとtestでも重複レコードあり
# - 欠損多い特徴量あり

# # EDA

# +
def check_out(df):
    # 外れ値確認
    num_cols = df.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"]).columns.to_list()
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=df[num_cols])
    plt.show()
    plt.clf()
    plt.close()
    
_df = df_all.copy().drop(["id", "y"], axis=1)
check_out(_df)


# -

# - A1-levelは外れ値ありそう

# + hide_input=false
def check_num_hist(df):
    # 数値列のヒストグラム。歪み（Skewness）大きい場合は対数変換した方が良い
    num_cols = df.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"]).columns.to_list()
    for col in num_cols:
        try:
            _ser = df[col].dropna()
            sns.distplot(_ser, label=f"Skewness: {round(_ser.skew(), 2)}")
            plt.legend()
            plt.show()
            plt.clf()
            plt.close()
        except Exception as e:
            print(e)
            print(f"ERROR: {col}")

_df = df_all.copy().drop(["id", "y"], axis=1)
check_num_hist(_df)


# -
# - A2-level, A3-level, A4-level, B1-level, B2-level, B3-level, B4-level は対数化した方が良いかも

# +
def log1p(df, num_cols):
    for col in num_cols:
        df[f"{col}_log1p"] = np.log(df[col] + 1)
    return df

log1p_cols = ["A2-level", "A3-level", "A4-level", "B1-level", "B2-level", "B3-level", "B4-level"]
df_all = log1p(df_all, log1p_cols)

new_log1p_cols = [f"{c}_log1p" for c in log1p_cols]
check_num_hist(df_all[new_log1p_cols])


# -

# - 歪度あまり改善しなので対数化しても意味なさそう

# +
def check_cate_count(df):
    # カテゴリ列のレコード数
    cate_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    for col in cate_cols:
        sns.countplot(x=col, data=df, color='salmon')
        plt.show()
        plt.clf()
        plt.close()
        
check_cate_count(df_all)
# -

# - 値がすべて同じ game-ver, lobbyは消しとくか

df_all = df_all.drop(["game-ver", "lobby"], axis=1)

# - trainとtestの期間はごちゃ混ぜか？

# +
train_df['period'] = pd.to_datetime(train_df['period'], format='%Y-%m-%d %H:%M:%S')
test_df['period'] = pd.to_datetime(test_df['period'], format='%Y-%m-%d %H:%M:%S')
df_all['period'] = pd.to_datetime(df_all['period'], format='%Y-%m-%d %H:%M:%S')

train_df = train_df.set_index('period', drop=False)
test_df = test_df.set_index('period', drop=False)
df_all = df_all.set_index('period', drop=False)

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

train_df = time_cols(train_df, 'period')
test_df = time_cols(test_df, 'period')
df_all = time_cols(df_all, 'period')

display(df_all.head(1))

# +
train_df["id"].plot(title="train_id", figsize=(12, 6))
plt.show()

test_df["id"].plot(title="test_df_id", figsize=(12, 6))
plt.show()

df_all["id"].plot(title="df_all_id", figsize=(12, 6))
plt.show()
# -

# - trainとtestの期間はごちゃ混ぜだった。cvは時系列のやり方で分けなくて良さそう

# カテゴリ列 vs. カテゴリ列。これは１列ごとに見てかないと理解できない
# countの場合は valuesはどの列選んでも同じ。columnsとindexが一意なレコードの数数える
str_cols = df_all.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
for col in str_cols:
    _df = df_all.pivot_table(values="id", index=col, columns="y", aggfunc="count",)
    _df.plot.bar(figsize=(12, 6))
    plt.show()
    plt.clf()
    plt.close()

# - rankによる勝ち負けの差は大きくはなさそう
# - 武器によって勝ち負けの差大きそう
# - mode違いで勝ち負けの差は少しありそう
# - ステージ違いで勝ち負けの差は大きくはなさそう

df_all["y"].value_counts()

# - ラベル不均衡でない

# 特徴量同士の相関ヒートマップ
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df_all.corr(), square=True, vmax=1, vmin=-1, center=0, annot=True)
plt.show()

# - 元々の列で相関明らかに高い列はない

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


# +
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
    agg_dict = {target_col: ["count", "mean"]}
    
    plot_config = {"bar": {"width": 0.8}}
    plot_target_stats(df_all, col, agg_dict, plot_config)


# -

# - レベル高いレコードは少ないので平均の値が極端
# - A1,A2,A3,A4,B1,B2,B3,B4それぞれレベルの分布が違う。A1はレベル高いレコード多めだが、ほかのはレベル高いのは少ない。雑魚狩りせずにA1に上がるから？

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

# # 特徴量追加

# データ再取得
train_df = pd.read_csv(f"{ORIG}/train_data.csv")
test_df = pd.read_csv(f"{ORIG}/test_data.csv")
df_all = train_df.append(test_df).reset_index(drop=True)


# +
# 上のセルでやった列追加削除

def time_cols(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col]) # dtype を datetime64 に変換
    # 期間短いから週と時間だけにしておく
    #df['year'] = df[date_col].dt.year
    #df['month'] = df[date_col].dt.month
    #df['day'] = df[date_col].dt.day
    #df['dayofyear'] = df[date_col].dt.dayofyear
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['weekend'] = (df[date_col].dt.dayofweek.values >= 5).astype(int)
    df['hour'] = df[date_col].dt.hour
    return df


def log1p(df, num_cols):
    for col in num_cols:
        df[f"{col}_log1p"] = np.log(df[col] + 1)
    return df


log1p_cols = ["A2-level", "A3-level", "A4-level", "B1-level", "B2-level", "B3-level", "B4-level"]
df_all = log1p(df_all, log1p_cols)

df_all['period'] = pd.to_datetime(df_all['period'], format='%Y-%m-%d %H:%M:%S')
df_all = df_all.set_index('period', drop=False)
df_all = time_cols(df_all, 'period')

# 不要な列削除
df_all = df_all.drop(["id", 'period', "game-ver", "lobby"], axis=1)
df_all


# +
def target_corr(df, target_col="y", png_path=None):
    """目的変数との数値列との相関係数確認"""
    num_cols = df.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"]).columns.to_list()
    if target_col in num_cols:
        num_cols.remove(target_col)
    corrs = []
    for col in num_cols:
        s1 = df[col]
        s2 = df[target_col]
        corr = s1.corr(s2)
        corrs.append(abs(round(corr, 3)))
        
    df_corr = pd.DataFrame({"feature": num_cols, "y_corr": corrs}).sort_values(by='y_corr', ascending=False)
    
    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="y_corr",
        y="feature",
        data=df_corr.head(50),
    )
    plt.title(f"target_corr")
    plt.tight_layout()
    if png_path is not None:
        plt.savefig(png_path)
        
    return df_corr
        
    
target_corr(df_all, target_col="y")
# -

# base_cate_cols = df_all.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
base_cate_cols = ["A1-rank", "A1-weapon", "A2-rank", "A2-weapon", "A3-rank", "A3-weapon", "A4-rank", "A4-weapon", 
                  "B1-rank", "B1-weapon", "B2-rank", "B2-weapon", "B3-rank", "B3-weapon", "B4-rank", "B4-weapon",
                  "lobby-mode", "mode", "stage",]


# +
def add_cate(df, cat_features):
    # 文字列を2列結合
    for col1, col2 in tqdm(itertools.combinations(cat_features, 2)):
        new_col_name = '_'.join([col1, col2])
        new_values = df[col1].map(str) + "_" + df[col2].map(str)
        #encoder = preprocessing.LabelEncoder()
        df[new_col_name] = new_values# encoder.fit_transform(new_values)

    return df

df_all = add_cate(df_all, base_cate_cols)
display(df_all)


# +
# A列でグループして集計したB列は意味がありそうと仮説たててから統計値列作ること
# 目的変数をキーにして集計するとリークしたターゲットエンコーディングになるため説明変数同士で行うこと
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

# 集計する数値列指定
value_agg = {
    'A1-level' : ['max', 'min', 'median', 'std', 'skew'],
    'A2-level' : ['max', 'min', 'median', 'std', 'skew'],
    'A3-level' : ['max', 'min', 'median', 'std', 'skew'],
    'A4-level' : ['max', 'min', 'median', 'std', 'skew'],
    'B1-level' : ['max', 'min', 'median', 'std', 'skew'],
    'B2-level' : ['max', 'min', 'median', 'std', 'skew'],
    'B3-level' : ['max', 'min', 'median', 'std', 'skew'],
    'B4-level' : ['max', 'min', 'median', 'std', 'skew'],
}
# グループ化するカテゴリ列でループ
for key in tqdm(base_cate_cols):
    feature_df = grouping(df_all, key, value_agg, prefix=key + "_")
    df_all = pd.merge(df_all, feature_df, how='left', on=key)    
display(df_all)


# +
def arithmetic_num_xfeat(df, operator="*", n_order=2, num_cols=None):
    """
    xfeatで数値列同士を算術計算
    operatorは「+,-,*」のいずれか
    n_orderは次数。2なら2列の組み合わせになる
    列膨大になるので掛け合わせる列num_colsで指定したほうがいい
    参考: https://megane-man666.hatenablog.com/entry/xfeat
    """
    if num_cols is None:
        df_num = Pipeline([SelectNumerical(),]).fit_transform(df)
        num_cols = df_num.columns.tolist()

    if operator == "+":
        output_suffix = "_plus"
    elif operator == "*":
        output_suffix = "_mul"
    elif operator == "-":
        output_suffix = "_minus"

    df = Pipeline(
        [
            ArithmeticCombinations(
                input_cols=num_cols,
                drop_origin=False,
                operator=operator,
                r=n_order,
                output_suffix=output_suffix,
            ),
        ]
    ).fit_transform(df)
    return df


base_num_cols = ["A1-level", "A2-level", "A3-level", "A4-level", "B1-level", "B2-level", "B3-level", "B4-level",]
df_all = arithmetic_num_xfeat(df_all, operator="*", num_cols=base_num_cols)
df_all = arithmetic_num_xfeat(df_all, operator="+", num_cols=base_num_cols)
df_all = arithmetic_num_xfeat(df_all, operator="-", num_cols=base_num_cols)

# ファイル出力
df_all.to_csv(f"{OUT_DATA}/feature_add.csv", index=False)
display(df_all)
# -

target_corr(df_all, target_col="y")

# # 特徴量選択

# +
df_all = pd.read_csv(f"{OUT_DATA}/feature_add.csv")

# label_encoding
cate_cols = df_all.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
for col in cate_cols:
    df_all[col], uni = pd.factorize(df_all[col])
    
# ファイル出力
#df_all.to_csv(f"{OUT_DATA}/label_encoding.csv", index=False)
display(df_all)


# +
def run_feature_selection(
    df, target_col, params,
):
    """feature_importanceの閾値固定して特徴量選択"""
    # 特徴量の列名（取捨選択前）
    input_cols = df.columns.tolist()
    n_before_selection = len(input_cols)
    input_cols.remove(target_col)

    # 特徴量選択用モデル取得
    lgbm_params = {
        "objective": params["objective"],
        "metric": params["metric"],
        "verbosity": -1,
    }
    selector = GBDTFeatureSelector(
        input_cols=input_cols,
        target_col=target_col,
        threshold=params["threshold"],
        lgbm_params=lgbm_params,
    )
    selector.fit(df)

    # 選択をした特徴量を返す
    selected_cols = selector.get_selected_cols()
    print(f" - {n_before_selection - len(selected_cols)} features are removed.")
    return df[selected_cols]


target_col = "y"
# metric=roc_aucでも可能
# feature_importance高い順に列数を 列数*threshold にする
params = {"metric": "binary_logloss", "objective": "binary", "threshold": 0.7}
df_all = run_feature_selection(df_all, target_col, params)
print(df_all.shape)
print(df_all.columns)

# 列名保持
feature_selections = sorted(df_all.columns.to_list())
pd.DataFrame({"feature_selections": feature_selections}).to_csv(f"{OUT_DATA}/feature_selections.csv", index=False)

# ファイル出力
df_all.to_csv(f"{OUT_DATA}/run_feature_selection.csv", index=False)
# -

for c in feature_selections:
    print(c)


# +
def remove_useless_features(df, cols=None, threshold=0.8):
    """
    xfeatで不要な特徴量削除
    - 列の内容が重複している列削除
    - すべて同じ値の列削除
    - スピマンの相関係数が高い列（多重共変性ある列）削除.相関係数がthresholdより高い列が消される
    https://github.com/pfnet-research/xfeat/blob/master/examples/remove_useless_features.py
    """
    # データ型を変換してメモリ使用量を削減
    cols = df.columns.tolist() if cols is None else cols
    df = compress_df(pd.DataFrame(data=df, columns=cols))

    encoder = Pipeline(
        [
            DuplicatedFeatureEliminator(),
            ConstantFeatureEliminator(),
            #SpearmanCorrelationEliminator(threshold=threshold),  # 相関係数>thresholdの特長削除
        ]
    )
    df_reduced = encoder.fit_transform(df)
    # print("Selected columns: {}".format(df_reduced.columns.tolist()))
    return df_reduced

y_seri = df_all["y"]
df_all = df_all.drop("y", axis=1)
df_all = remove_useless_features(df_all)
df_all = pd.concat([df_all, y_seri], axis=1)
display(df_all)
df_all.to_csv(f"{OUT_DATA}/remove_useless_features.csv")


# +
def plot_rfecv(selector):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()

    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    plt.show()
    plt.clf()
    plt.close()


X_train = df_all.loc[df_all["y"].notnull()]
cate_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
for col in cate_cols:
    X_train[col], uni = pd.factorize(X_train[col])
y_train = X_train["y"]

clf = lgb.LGBMClassifier(n_jobs=-1, seed=71)  # 欠損ある場合はGBM使う（リッジより遅い）

# RFECVは交差検証+再帰的特徴除去。データでかいとメモリ死ぬので注意
# RFE（再帰的特徴除去=recursive feature elimination: すべての特徴量を使う状態から、1つずつ特徴量を取り除いていく）で特徴量選択
selector = RFECV(clf, cv=KFold(3, shuffle=True), scoring="accuracy", n_jobs=-1)
selector.fit(X_train, y_train)

# 選択した特徴量
select_cols = X_train.columns[selector.get_support()].to_list()
print("\nselect_cols:\n", select_cols)
# 捨てた特徴量
print("not select_cols:\n", X_train.columns[~selector.get_support()].to_list())
plot_rfecv(selector)
df_all.to_csv(f"{OUT_DATA}/rfecv.csv")


# -
# # モデル作成

# +
def count_encoder(train_df, valid_df, cat_features=None):
    """
    Count_Encoding: カテゴリ列をカウント値に変換する特徴量エンジニアリング（要はgroupby().size()の集計列追加のこと）
    ※カウント数が同じカテゴリは同じようなデータ傾向になる可能性がある
    https://www.kaggle.com/matleonard/categorical-encodings
    """
    # conda install -c conda-forge category_encoders
    import category_encoders as ce
    
    if cat_features is None:
        cat_features = train_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()

    count_enc = ce.CountEncoder(cols=cat_features)

    # trainだけでfitすること(validationやtest含めるとリークする)
    count_enc.fit(train_df[cat_features])
    train_encoded = train_df.join(count_enc.transform(train_df[cat_features]).add_suffix("_count"))
    valid_encoded = valid_df.join(count_enc.transform(valid_df[cat_features]).add_suffix("_count"))
    
    return train_encoded, valid_encoded


def target_encoder(train_df, valid_df, target_col:str, cat_features=None):
    """
    Target_Encoding: カテゴリ列を目的変数の平均値に変換する特徴量エンジニアリング
    https://www.kaggle.com/matleonard/categorical-encodings
    """
    # conda install -c conda-forge category_encoders
    import category_encoders as ce

    if cat_features is None:
        cat_features = train_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    
    target_enc = ce.TargetEncoder(cols=cat_features)

    # trainだけでfitすること(validationやtest含めるとリークする)
    target_enc.fit(train_df[cat_features], train_df[target_col])

    train_encoded = train_df.join(target_enc.transform(train_df[cat_features]).add_suffix("_target"))
    valid_encoded = valid_df.join(target_enc.transform(valid_df[cat_features]).add_suffix("_target"))
    return train_encoded, valid_encoded


def catboost_encoder(train_df, valid_df, target_col:str, cat_features=None):
    """
    CatBoost_Encoding: カテゴリ列を目的変数の1行前の行からのみに変換する特徴量エンジニアリング
    CatBoost使ったターゲットエンコーディング
    https://www.kaggle.com/matleonard/categorical-encodings
    """
    # conda install -c conda-forge category_encoders
    import category_encoders as ce

    if cat_features is None:
        cat_features = train_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    
    cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)

    # trainだけでfitすること(validationやtest含めるとリークする)
    cb_enc.fit(train_df[cat_features], train_df[target_col])

    train_encoded = train_df.join(cb_enc.transform(train_df[cat_features]).add_suffix("_cb"))
    valid_encoded = valid_df.join(cb_enc.transform(valid_df[cat_features]).add_suffix("_cb"))
    return train_encoded, valid_encoded


# + code_folding=[113, 161, 166, 179]
class Model:
    def __init__(self, OUTPUT_DIR):
        self.OUTPUT_DIR = OUTPUT_DIR

    # LightGBM GBDT with KFold or Stratified KFold
    # Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
    def kfold_cv_LGBMClassifier(
        self,
        lgb_params: dict,
        df: pd.DataFrame,
        num_folds: int,
        target_col: str,
        del_cols=None,
        eval_metric="error",
        stratified=True,  # StratifiedKFoldにするか
        is_submission=False,  # Home_Credit_Default_Risk の submission.csv作成するか
        is_plot_perm_importance=False,  # permutation importanceも出すか. feature_importance はデフォルトでだす
    ):
        """
        LGBMClassifierでcross validation + feature_importance/permutation importance plot
        """
        # Divide in training/validation and test data
        train_df = df[df[target_col].notnull()].reset_index(drop=True)
        test_df = df[df[target_col].isnull()].reset_index(drop=True)
        print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
        del df
        gc.collect()
        
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
        else:
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

        # Create arrays and dataframes to store results
        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        feature_importance_df = pd.DataFrame()
        permutation_importance_df = pd.DataFrame()
        result_scores = {}

        # 目的変数とID列など削除
        del_cols = del_cols.append(target_col) if del_cols is not None else [target_col]
        feats = [f for f in train_df.columns if f not in del_cols]

        for n_fold, (train_idx, valid_idx) in tqdm(
            enumerate(folds.split(train_df[feats], train_df[target_col]))
        ):
            t_fold_df = train_df.iloc[train_idx]
            v_fold_df = train_df.iloc[valid_idx]
            
            # カウントエンコディング
            t_fold_df, v_fold_df = count_encoder(t_fold_df, v_fold_df, cat_features=None)
            # ターゲットエンコディング
            t_fold_df, v_fold_df = target_encoder(t_fold_df, v_fold_df, target_col=target_col, cat_features=None)
            # CatBoostエンコディング
            t_fold_df, v_fold_df = catboost_encoder(t_fold_df, v_fold_df, target_col=target_col, cat_features=None)
            # ラベルエンコディング
            cate_cols = t_fold_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
            for col in cate_cols:
                t_fold_df[col], uni = pd.factorize(t_fold_df[col])
                v_fold_df[col], uni = pd.factorize(v_fold_df[col])
            print("run encoding Train shape: {}, valid shape: {}".format(t_fold_df.shape, v_fold_df.shape))
            
            feats = t_fold_df.columns.to_list()
            feats.remove(target_col)
            
            train_x, train_y = (
                t_fold_df[feats],
                t_fold_df[target_col],
            )
            valid_x, valid_y = (
                v_fold_df[feats],
                v_fold_df[target_col],
            )
            
            # LightGBM parameters found by Bayesian optimization
            clf = LGBMClassifier(**lgb_params)

            clf.fit(
                train_x,
                train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric=eval_metric,
                verbose=200,
                early_stopping_rounds=200,
            )

            # モデル保存
            joblib.dump(clf, f"{self.OUTPUT_DIR}/lgb-{n_fold + 1}.model", compress=True)
            
            # valid pred
            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            
            # test pred
            # カウントエンコディング
            tr_df, te_df = count_encoder(train_df, test_df, cat_features=None)
            # ターゲットエンコディング
            tr_df, te_df = target_encoder(tr_df, te_df, target_col=target_col, cat_features=None)
            # CatBoostエンコディング
            tr_df, te_df = catboost_encoder(tr_df, te_df, target_col=target_col, cat_features=None)
            # ラベルエンコディング
            cate_cols = tr_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
            for col in cate_cols:
                tr_df[col], uni = pd.factorize(tr_df[col])
                te_df[col], uni = pd.factorize(te_df[col])
            sub_preds += clf.predict_proba(te_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
            
            if eval_metric == "auc":
                fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
                print("Fold %2d AUC : %.6f" % (n_fold + 1, fold_auc))
                result_scores[f"fold_auc_{str(n_fold + 1)}"] = fold_auc
            elif eval_metric == "error":
                # intにしないとaccuracy_score()エラーになる
                _pred = oof_preds[valid_idx]
                _pred[_pred >= 0.5] = 1
                _pred[_pred < 0.5] = 0
                fold_err = 1.0 - accuracy_score(valid_y, _pred)
                print("Fold %2d error : %.6f" % (n_fold + 1, fold_err))
                result_scores[f"fold_err_{str(n_fold + 1)}"] = fold_err

            # feature_importance
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            if is_plot_perm_importance:
                # permutation_importance
                # 時間かかるからifで制御する
                # scoringはsklearnのスコアリングパラメータ
                # accuracy や neg_mean_squared_log_error とか
                # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                fold_importance_df = pd.DataFrame()
                fold_permutation = permutation_importance(
                    clf, valid_x, valid_y, scoring="roc_auc"
                )
                fold_permutation_df = pd.DataFrame(
                    {
                        "feature": valid_x.columns,
                        "importance": np.abs(
                            fold_permutation["importances_mean"]
                        ),  # マイナスとるのもあるので絶対値にする
                        "fold": n_fold + 1,
                    },
                )
                permutation_importance_df = pd.concat(
                    [permutation_importance_df, fold_permutation_df], axis=0
                )

            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        if eval_metric == "auc":
            mean_fold_auc = roc_auc_score(train_df[target_col], oof_preds)
            print("Full AUC score %.6f" % mean_fold_auc)
            result_scores["mean_fold_auc"] = mean_fold_auc
        elif eval_metric == "error":
            # intにしないとaccuracy_score()エラーになる
            _pred = oof_preds
            _pred[_pred >= 0.5] = 1
            _pred[_pred < 0.5] = 0
            mean_fold_err = 1.0 - accuracy_score(train_df[target_col], _pred)
            print("Full error score %.6f" % mean_fold_err)
            result_scores["mean_fold_err"] = mean_fold_err

        # モデルのスコア出力
        result_scores_df = pd.DataFrame(result_scores.values(), index=result_scores.keys())
        result_scores_df.to_csv(f"{self.OUTPUT_DIR}/result_scores.tsv", sep="\t")

        # Write submission file (Home_Credit_Default_Risk)
        if is_submission:
            sub_preds[sub_preds >= 0.5] = 1
            sub_preds[sub_preds < 0.5] = 0
            test_df[target_col] = sub_preds
            submission_file_name = f"{self.OUTPUT_DIR}/submission_kernel.csv"
            sub_df = test_df[[target_col]]
            sub_df["id"] = test_df.index
            sub_df.astype(int)
            sub_df = sub_df[["id", "y"]]
            sub_df.to_csv(submission_file_name, index=False)

        # Plot feature importance
        Model("").display_importances(
            feature_importance_df,
            png_path=f"{self.OUTPUT_DIR}/lgbm_feature_importances.png",
            title="feature_importance",
        )
        if is_plot_perm_importance:
            Model("").display_importances(
                permutation_importance_df,
                png_path=f"{self.OUTPUT_DIR}/lgbm_permutation_importances.png",
                title="permutation_importance",
            )

        return feature_importance_df, permutation_importance_df

    # Display/plot feature/permutation importance
    @staticmethod
    def display_importances(
        importance_df_, png_path, title,
    ):
        cols = (
            importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:40]
            .index
        )
        best_features = importance_df_.loc[importance_df_.feature.isin(cols)]
        plt.figure(figsize=(8, 10))
        sns.barplot(
            x="importance",
            y="feature",
            data=best_features.sort_values(by="importance", ascending=False),
        )
        plt.title(f"LightGBM {title} (avg over folds)")
        plt.tight_layout()
        plt.savefig(png_path)


if __name__ == "__main__":
    df = df_all
    
    #df_all = pd.read_csv(f"{OUT_DATA}/feature_add.csv")
    #cols = pd.read_csv(f"{OUT_DATA}/feature_selections.csv")["feature_selections"].values.tolist()
    #cols.append("y")
    #df = df_all[cols]
    
    is_debug = False
    #is_debug = True
    if is_debug:
        _df = df.head(1000)
        _df = _df.append(df.tail(100))
        df = _df
    
    model = Model(OUT_MODEL)
    lgb_params = dict(
        n_estimators=10000,
        learning_rate=0.01,
        silent=-1,
        verbose=-1,
        # verbose=1,
        importance_type="gain",
    )
    params = dict(
        lgb_params=lgb_params,
        df=df,
        num_folds=10,
        target_col="y",
        del_cols=None,
        eval_metric="error",
        stratified=True,
        is_submission=True,
        is_plot_perm_importance=False,
    )
    feat_importance, perm_importance = model.kfold_cv_LGBMClassifier(**params)
# -
# # パラメータチューニング

# + code_folding=[107, 112, 116, 132]
import sys
sys.path.append(r'C:\Users\81908\Git\OptGBM')
import optgbm as lgb

class Model:
    def __init__(self, OUTPUT_DIR):
        self.OUTPUT_DIR = OUTPUT_DIR

    # LightGBM GBDT with KFold or Stratified KFold
    # Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
    def kfold_cv_LGBMClassifier(
        self,
        lgb_params: dict,
        df: pd.DataFrame,
        num_folds: int,
        target_col: str,
        del_cols=None,
        eval_metric="error",
        stratified=True,  # StratifiedKFoldにするか
        is_submission=False,  # Home_Credit_Default_Risk の submission.csv作成するか
        is_plot_perm_importance=False,  # permutation importanceも出すか. feature_importance はデフォルトでだす
    ):
        """
        LGBMClassifierでcross validation + feature_importance/permutation importance plot
        """
        # Divide in training/validation and test data
        train_df = df[df[target_col].notnull()].reset_index(drop=True)
        test_df = df[df[target_col].isnull()].reset_index(drop=True)
        print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
        del df
        gc.collect()
        
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
        else:
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

        # Create arrays and dataframes to store results
        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        feature_importance_df = pd.DataFrame()
        permutation_importance_df = pd.DataFrame()
        result_scores = {}

        # 目的変数とID列など削除
        del_cols = del_cols.append(target_col) if del_cols is not None else [target_col]
        feats = [f for f in train_df.columns if f not in del_cols]

        for n_fold, (train_idx, valid_idx) in tqdm(
            enumerate(folds.split(train_df[feats], train_df[target_col]))
        ):
            t_fold_df = train_df.iloc[train_idx]
            v_fold_df = train_df.iloc[valid_idx]
            
            # カウントエンコディング
            t_fold_df, v_fold_df = count_encoder(t_fold_df, v_fold_df, cat_features=None)
            # ターゲットエンコディング
            t_fold_df, v_fold_df = target_encoder(t_fold_df, v_fold_df, target_col=target_col, cat_features=None)
            # CatBoostエンコディング
            t_fold_df, v_fold_df = catboost_encoder(t_fold_df, v_fold_df, target_col=target_col, cat_features=None)
            # ラベルエンコディング
            cate_cols = t_fold_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
            for col in cate_cols:
                t_fold_df[col], uni = pd.factorize(t_fold_df[col])
                v_fold_df[col], uni = pd.factorize(v_fold_df[col])
            print("run encoding Train shape: {}, valid shape: {}".format(t_fold_df.shape, v_fold_df.shape))
            
            feats = t_fold_df.columns.to_list()
            feats.remove(target_col)
            
            train_x, train_y = (
                t_fold_df[feats],
                t_fold_df[target_col],
            )
            valid_x, valid_y = (
                v_fold_df[feats],
                v_fold_df[target_col],
            )
            
            # LightGBM parameters found by Bayesian optimization
            clf = lgb.LGBMClassifier(**lgb_params)

            clf.fit(
                train_x,
                train_y,
                #eval_set=[(train_x, train_y), (valid_x, valid_y)],
                #eval_metric=eval_metric,
                #verbose=200,
                #early_stopping_rounds=200,
            )

            # モデル保存
            joblib.dump(clf, f"{self.OUTPUT_DIR}/lgb-{n_fold + 1}.model", compress=True)
            
            # valid pred
            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            
            # test pred
            # カウントエンコディング
            tr_df, te_df = count_encoder(train_df, test_df, cat_features=None)
            # ターゲットエンコディング
            tr_df, te_df = target_encoder(tr_df, te_df, target_col=target_col, cat_features=None)
            # CatBoostエンコディング
            tr_df, te_df = catboost_encoder(tr_df, te_df, target_col=target_col, cat_features=None)
            # ラベルエンコディング
            cate_cols = tr_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
            for col in cate_cols:
                tr_df[col], uni = pd.factorize(tr_df[col])
                te_df[col], uni = pd.factorize(te_df[col])
            sub_preds += clf.predict_proba(te_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
            
            if eval_metric == "auc":
                fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
                print("Fold %2d AUC : %.6f" % (n_fold + 1, fold_auc))
                result_scores[f"fold_auc_{str(n_fold + 1)}"] = fold_auc
            elif eval_metric == "error":
                # intにしないとaccuracy_score()エラーになる
                _pred = oof_preds[valid_idx]
                _pred[_pred >= 0.5] = 1
                _pred[_pred < 0.5] = 0
                fold_err = 1.0 - accuracy_score(valid_y, _pred)
                print("Fold %2d error : %.6f" % (n_fold + 1, fold_err))
                result_scores[f"fold_err_{str(n_fold + 1)}"] = fold_err

            # feature_importance
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            if is_plot_perm_importance:
                # permutation_importance
                # 時間かかるからifで制御する
                # scoringはsklearnのスコアリングパラメータ
                # accuracy や neg_mean_squared_log_error とか
                # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                fold_importance_df = pd.DataFrame()
                fold_permutation = permutation_importance(
                    clf, valid_x, valid_y, scoring="roc_auc"
                )
                fold_permutation_df = pd.DataFrame(
                    {
                        "feature": valid_x.columns,
                        "importance": np.abs(
                            fold_permutation["importances_mean"]
                        ),  # マイナスとるのもあるので絶対値にする
                        "fold": n_fold + 1,
                    },
                )
                permutation_importance_df = pd.concat(
                    [permutation_importance_df, fold_permutation_df], axis=0
                )

            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        if eval_metric == "auc":
            mean_fold_auc = roc_auc_score(train_df[target_col], oof_preds)
            print("Full AUC score %.6f" % mean_fold_auc)
            result_scores["mean_fold_auc"] = mean_fold_auc
        elif eval_metric == "error":
            # intにしないとaccuracy_score()エラーになる
            _pred = oof_preds
            _pred[_pred >= 0.5] = 1
            _pred[_pred < 0.5] = 0
            mean_fold_err = 1.0 - accuracy_score(train_df[target_col], _pred)
            print("Full error score %.6f" % mean_fold_err)
            result_scores["mean_fold_err"] = mean_fold_err

        # モデルのスコア出力
        result_scores_df = pd.DataFrame(result_scores.values(), index=result_scores.keys())
        result_scores_df.to_csv(f"{self.OUTPUT_DIR}/result_scores.tsv", sep="\t")

        # Write submission file (Home_Credit_Default_Risk)
        if is_submission:
            sub_preds[sub_preds >= 0.5] = 1
            sub_preds[sub_preds < 0.5] = 0
            test_df[target_col] = sub_preds
            submission_file_name = f"{self.OUTPUT_DIR}/submission_kernel.csv"
            sub_df = test_df[[target_col]]
            sub_df["id"] = test_df.index
            sub_df.astype(int)
            sub_df = sub_df[["id", "y"]]
            sub_df.to_csv(submission_file_name, index=False)

        # Plot feature importance
        Model("").display_importances(
            feature_importance_df,
            png_path=f"{self.OUTPUT_DIR}/lgbm_feature_importances.png",
            title="feature_importance",
        )
        if is_plot_perm_importance:
            Model("").display_importances(
                permutation_importance_df,
                png_path=f"{self.OUTPUT_DIR}/lgbm_permutation_importances.png",
                title="permutation_importance",
            )

        return feature_importance_df, permutation_importance_df

    # Display/plot feature/permutation importance
    @staticmethod
    def display_importances(
        importance_df_, png_path, title,
    ):
        cols = (
            importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:40]
            .index
        )
        best_features = importance_df_.loc[importance_df_.feature.isin(cols)]
        plt.figure(figsize=(8, 10))
        sns.barplot(
            x="importance",
            y="feature",
            data=best_features.sort_values(by="importance", ascending=False),
        )
        plt.title(f"LightGBM {title} (avg over folds)")
        plt.tight_layout()
        plt.savefig(png_path)


if __name__ == "__main__":
    df = df_all
    
    #df_all = pd.read_csv(f"{OUT_DATA}/feature_add.csv")
    #cols = pd.read_csv(f"{OUT_DATA}/feature_selections.csv")["feature_selections"].values.tolist()
    #cols.append("y")
    #df = df_all[cols]
    
    is_debug = False
    #is_debug = True
    if is_debug:
        _df = df.head(1000)
        _df = _df.append(df.tail(100))
        df = _df
    
    model = Model(OUT_MODEL)
    lgb_params = dict(
        n_estimators=10000,
        learning_rate=0.1,
        silent=-1,
        verbose=-1,
        # verbose=1,
        importance_type="gain",
    )
    params = dict(
        lgb_params=lgb_params,
        df=df,
        num_folds=10,
        target_col="y",
        del_cols=None,
        eval_metric="error",
        stratified=True,
        is_submission=True,
        is_plot_perm_importance=False,
    )
    feat_importance, perm_importance = model.kfold_cv_LGBMClassifier(**params)
# -











# +
import optgbm as lgb
from sklearn.datasets import load_boston

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df = df.drop(["alive"], axis=1)
for col in ["sex", "embarked", "who", "embark_town", "class", "adult_male", "alone", "deck"]:
    df[col], uni = pd.factorize(df[col])
df.iloc[700:, 0] = np.nan  # test set用意
y = df["survived"]
X = df.drop("survived", axis=1)
display(df)

reg = lgb.LGBMClassifier(random_state=0)
reg.fit(X, y)
score = reg.score(X, y)
# -




























# +

X_train
# -












from xfeat.utils import compress_df







df_all.select_dtypes(
    include=["int", "int32", "int64", "float", "float32", "float64"]
).columns.to_list()

df_all.select_dtypes(include=["object", "category", "bool"]).columns.to_list()






















