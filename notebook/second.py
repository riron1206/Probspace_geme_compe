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
pd.set_option("display.max_columns", None)

# +
OUT_DATA = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\second"
os.makedirs(OUT_DATA, exist_ok=True)

OUT_MODEL = (
    r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\model\second"
)
os.makedirs(OUT_MODEL, exist_ok=True)

ORIG = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\orig"
train_df = pd.read_csv(f"{ORIG}/train_data.csv")
test_df = pd.read_csv(f"{ORIG}/test_data.csv")
df_all = train_df.append(test_df).reset_index(drop=True)


# -

# helper関数


def target_corr(df, target_col="y", png_path=None):
    """目的変数との数値列との相関係数確認"""
    num_cols = df.select_dtypes(
        include=["int", "int32", "int64", "float", "float32", "float64"]
    ).columns.to_list()
    if target_col in num_cols:
        num_cols.remove(target_col)
    corrs = []
    for col in num_cols:
        s1 = df[col]
        s2 = df[target_col]
        corr = s1.corr(s2)
        corrs.append(abs(round(corr, 3)))

    df_corr = pd.DataFrame({"feature": num_cols, "y_corr": corrs}).sort_values(
        by="y_corr", ascending=False
    )

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="y_corr", y="feature", data=df_corr.head(50),
    )
    plt.title(f"target_corr")
    plt.tight_layout()
    if png_path is not None:
        plt.savefig(png_path)

    return df_corr


# # 特徴量追加

# +
def time_cols(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])  # dtype を datetime64 に変換
    # 期間短いから週と時間だけにしておく
    # df['year'] = df[date_col].dt.year
    # df['month'] = df[date_col].dt.month
    # df['day'] = df[date_col].dt.day
    # df['dayofyear'] = df[date_col].dt.dayofyear
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["weekend"] = (df[date_col].dt.dayofweek.values >= 5).astype(int)
    df["hour"] = df[date_col].dt.hour
    return df


df_all["period"] = pd.to_datetime(df_all["period"], format="%Y-%m-%d %H:%M:%S")
df_all = df_all.set_index("period", drop=False)
df_all = time_cols(df_all, "period")
print(df_all.shape)

# +
# 不要な列削除
df_all = df_all.drop(["id", "period", "game-ver", "lobby"], axis=1)

base_num_cols = df_all.select_dtypes(
    include=["int", "int32", "int64", "float", "float32", "float64"]
).columns.to_list()
base_cate_cols = df_all.select_dtypes(
    include=["object", "category", "bool"]
).columns.to_list()
print(df_all.shape)


# +
def log1p(df, num_cols):
    for col in num_cols:
        df[f"{col}_log1p"] = np.log(df[col] + 1)
    return df


log1p_cols = [
    "A2-level",
    "A3-level",
    "A4-level",
    "B1-level",
    "B2-level",
    "B3-level",
    "B4-level",
]
df_all = log1p(df_all, log1p_cols)
print(df_all.shape)


# +
def add_cate(df, cat_features):
    # 文字列を2列結合
    for col1, col2 in tqdm(itertools.combinations(cat_features, 2)):
        new_col_name = "_".join([col1, col2])
        new_values = df[col1].map(str) + "_" + df[col2].map(str)
        # encoder = preprocessing.LabelEncoder()
        df[new_col_name] = new_values  # encoder.fit_transform(new_values)

    return df

# rankと武器のペアで列作成
for i in range(1,5):
    df_all = add_cate(df_all, [f"A{i}-rank", f"A{i}-weapon"])
    df_all = add_cate(df_all, [f"B{i}-rank", f"B{i}-weapon"])
# display(df_all)
print(df_all.shape)


# +
def add_cates(df, cols, new_col_name):
    """複数列一気に連結"""
    df[new_col_name] = df[cols[0]].str.cat(df[cols[1:]], sep='_')
    return df


# Aチーム 4人 vs Bチーム 4人で戦ってるっぽいからA毎、B毎に組み合わせる
a_ranks = [
    "A1-rank",
    "A2-rank",
    "A3-rank",
    "A4-rank",
]
a_weapons = [
    "A1-weapon",
    "A2-weapon",
    "A3-weapon",
    "A4-weapon",
]
b_ranks = [
    "B1-rank",
    "B2-rank",
    "B3-rank",
    "B4-rank",
]
b_weapons = [
    "B1-weapon",
    "B2-weapon",
    "B3-weapon",
    "B4-weapon",
]

df_all = add_cates(df_all, a_ranks, "A_ranks")
df_all = add_cates(df_all, a_weapons, "A_weapons")
df_all = add_cates(df_all, [*a_ranks, *a_weapons], "A_ranks_weapons")
df_all = add_cates(df_all, b_ranks, "B_ranks")
df_all = add_cates(df_all, b_weapons, "B_weapons")
df_all = add_cates(df_all, [*b_ranks, *b_weapons], "B_ranks_weapons")

# display(df_all)
print(df_all.shape)


# + code_folding=[20, 182, 207, 215]
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


class AggUtil:
    ############## カテゴリ列 vs. 数値列について ##############
    @staticmethod
    def percentile(n):
        """パーセンタイル"""

        def percentile_(x):
            return np.percentile(x, n)

        percentile_.__name__ = "percentile_%s" % n
        return percentile_

    @staticmethod
    def diff_percentile(n1, n2):
        """パーセンタイルの差"""

        def diff_percentile_(x):
            p1 = np.percentile(x, n1)
            p2 = np.percentile(x, n2)
            return p1 - p2

        diff_percentile_.__name__ = f"diff_percentile_{n1}-{n2}"
        return diff_percentile_

    @staticmethod
    def ratio_percentile(n1, n2):
        """パーセンタイルの比"""

        def ratio_percentile_(x):
            p1 = np.percentile(x, n1)
            p2 = np.percentile(x, n2)
            return p1 / p2

        ratio_percentile_.__name__ = f"ratio_percentile_{n1}-{n2}"
        return ratio_percentile_

    @staticmethod
    def mean_var():
        """平均分散"""

        def mean_var_(x):
            x = x.dropna()
            return np.std(x) / np.mean(x)

        mean_var_.__name__ = f"mean_var"
        return mean_var_

    @staticmethod
    def diff_mean():
        """平均との差の中央値(aggは集計値でないとエラーになるから中央値をとる)"""

        def diff_mean_(x):
            x = x.dropna()
            return np.median(x - np.mean(x))

        diff_mean_.__name__ = f"diff_mean"
        return diff_mean_

    @staticmethod
    def ratio_mean():
        """平均との比の中央値(aggは一意な値でないとエラーになるから中央値をとる)"""

        def ratio_mean_(x):
            x = x.dropna()
            return np.median(x / np.mean(x))

        ratio_mean_.__name__ = f"ratio_mean"
        return ratio_mean_

    @staticmethod
    def hl_ratio():
        """平均より高いサンプル数と低いサンプル数の比率"""

        def hl_ratio_(x):
            x = x.dropna()
            n_high = x[x >= np.mean(x)].shape[0]
            n_low = x[x < np.mean(x)].shape[0]
            if n_low == 0:
                return 1.0
            else:
                return n_high / n_low

        hl_ratio_.__name__ = f"hl_ratio"
        return hl_ratio_

    @staticmethod
    def ratio_range():
        """最大/最小"""

        def ratio_range_(x):
            x = x.dropna()
            if np.min(x) == 0:
                return 1.0
            else:
                return np.max(x) / np.min(x)

        ratio_range_.__name__ = f"ratio_range"
        return ratio_range_

    @staticmethod
    def beyond1std():
        """1stdを超える比率"""

        def beyond1std_(x):
            x = x.dropna()
            if x.shape[0] == 0:
                return 1.0
            else:
                return x[np.abs(x) > np.abs(np.std(x))].shape[0] / x.shape[0]

        beyond1std_.__name__ = "beyond1std"
        return beyond1std_

    @staticmethod
    def zscore():
        """Zスコアの中央値(aggは一意な値でないとエラーになるから中央値をとる)"""

        def zscore_(x):
            x = x.dropna()
            return np.median((x - np.mean(x)) / np.std(x))

        zscore_.__name__ = "zscore"
        return zscore_

    ######################################################

    ############## カテゴリ列 vs. カテゴリ列について ##############
    @staticmethod
    def freq_entropy():
        """出現頻度のエントロピー"""
        from scipy.stats import entropy

        def freq_entropy_(x):
            return entropy(x.value_counts().values)

        freq_entropy_.__name__ = "freq_entropy"
        return freq_entropy_

    @staticmethod
    def freq1name():
        """最も頻繁に出現するカテゴリの数"""

        def freq1name_(x):
            return x.value_counts().sort_values(ascending=False)[0]

        freq1name_.__name__ = "freq1name"
        return freq1name_

    @staticmethod
    def freq1ratio():
        """最も頻繁に出現するカテゴリ/グループの数"""

        def freq1ratio_(x):
            frq = x.value_counts().sort_values(ascending=False)
            return frq[0] / frq.shape[0]

        freq1ratio_.__name__ = "freq1ratio"
        return freq1ratio_

    #########################################################


num_aggs = [
    "max",
    "min",
    "sum",
    "mean",
    "median",
    "prod",  # 積
    "var",  # 分散
    "std",  # 標準偏差
    "sem",  # 標準誤差
    "mad",  # 中央値の絶対偏差
    "skew",  # 歪度
    pd.DataFrame.kurt,  # 尖度
    np.ptp,  # peak to peak: 最大値と最小値との差
    AggUtil().percentile(25),  # 25パーセンタイル
    AggUtil().percentile(75),  # 75パーセンタイル
    AggUtil().diff_percentile(75, 25),  # パーセンタイルの差
    AggUtil().mean_var(),  # 平均分散
    AggUtil().diff_mean(),  # 平均との差の中央値
    AggUtil().ratio_mean(),  # 平均との比の中央値
    AggUtil().ratio_range(),  # 最大/最小
    AggUtil().hl_ratio(),  # 平均より高いサンプル数と低いサンプル数の比率
    AggUtil().beyond1std(),  # 1stdを超える比率
    AggUtil().zscore(),  # Zスコアの中央値
]
cate_aggs = [
    "count",  # カウント数
    AggUtil().freq_entropy(),  # 出現頻度のエントロピー
    AggUtil().freq1name(),  # 最も頻繁に出現するカテゴリの数
    AggUtil().freq1ratio(),  # 最も頻繁に出現するカテゴリ/グループの数
]

# 集計する数値列指定
num_value_agg = {
    "A1-level": num_aggs,
    "A2-level": num_aggs,
    "A3-level": num_aggs,
    "A4-level": num_aggs,
    "B1-level": num_aggs,
    "B2-level": num_aggs,
    "B3-level": num_aggs,
    "B4-level": num_aggs,
}
# グループ化するカテゴリ列でループ
keys = df_all.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
for key in tqdm(keys):
    #print(key, num_value_agg)
    feature_df = grouping(df_all, key, num_value_agg, prefix=key + "_")
    df_all = pd.merge(df_all, feature_df, how="left", on=key)

# 集計するカテゴリ列指定
keys = ["stage"]  # "lobby-mode", "mode", はなぜかエラーになる
cols = df_all.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
cate_value_agg = {}
for col in cols:
    if col not in keys:
        cate_value_agg[col] = cate_aggs
# グループ化するカテゴリ列でループ
for key in tqdm(keys):
    #print(key, cate_value_agg)
    feature_df = grouping(df_all, key, cate_value_agg, prefix=key + "_")
    df_all = pd.merge(df_all, feature_df, how="left", on=key)
    
pd.set_option("display.max_columns", None)
# display(df_all)
print(df_all.shape)


# + code_folding=[0]
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


# Aチーム 4人 vs Bチーム 4人で戦ってるっぽいからA毎、B毎に組み合わせる
a_num_cols = [
    "A1-level",
    "A2-level",
    "A3-level",
    "A4-level",
]
b_num_cols = [
    "B1-level",
    "B2-level",
    "B3-level",
    "B4-level",
]
for num_cols in [a_num_cols, b_num_cols]:
    df_all = arithmetic_num_xfeat(df_all, operator="*", n_order=4, num_cols=num_cols)
    df_all = arithmetic_num_xfeat(df_all, operator="+", n_order=4, num_cols=num_cols)
    df_all = arithmetic_num_xfeat(df_all, operator="-", n_order=4, num_cols=num_cols)
    
display(df_all)

# +
# ファイル出力
df_all.to_csv(f"{OUT_DATA}/feature_add.csv", index=False)
# display(df_all)

# 目的変数と数値列について相関係数確認
target_corr(df_all, target_col="y")
# -

# # 特徴量選択

# +
df_all = pd.read_csv(f"{OUT_DATA}/feature_add.csv")

# label_encoding
cate_cols = df_all.select_dtypes(
    include=["object", "category", "bool"]
).columns.to_list()
for col in cate_cols:
    df_all[col], uni = pd.factorize(df_all[col])

# ファイル出力
# df_all.to_csv(f"{OUT_DATA}/label_encoding.csv", index=False)
# display(df_all)
print(df_all.shape)


# + code_folding=[0]
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
            # SpearmanCorrelationEliminator(threshold=threshold),  # 相関係数>thresholdの特長削除
        ]
    )
    df_reduced = encoder.fit_transform(df)
    # print("Selected columns: {}".format(df_reduced.columns.tolist()))
    return df_reduced


y_seri = df_all["y"]
df_all = df_all.drop("y", axis=1)
df_all = remove_useless_features(df_all)
df_all = pd.concat([df_all, y_seri], axis=1)
# display(df_all)
df_all.to_csv(f"{OUT_DATA}/remove_useless_features.csv")


# + code_folding=[2]
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
n_cols = 600
threshold = n_cols / df_all.shape[1]
params = {"metric": "binary_logloss", "objective": "binary", "threshold": threshold}
df_all = run_feature_selection(df_all, target_col, params)
print(df_all.shape)
print(df_all.columns)

# 列名保持
feature_selections = sorted(df_all.columns.to_list())
pd.DataFrame({"feature_selections": feature_selections}).to_csv(
    f"{OUT_DATA}/feature_selections.csv", index=False
)

# ファイル出力
df_all.to_csv(f"{OUT_DATA}/run_feature_selection.csv", index=False)


# + code_folding=[]
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


df_all = pd.read_csv(f"{OUT_DATA}/feature_add.csv")
cols = pd.read_csv(f"{OUT_DATA}/feature_selections.csv")["feature_selections"].values.tolist()
target_col = "y"
cols.append(target_col)
df_all = df_all[cols]   
print(df_all.shape)
    
X_train = df_all.loc[df_all["y"].notnull()]
cate_cols = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.to_list()
for col in cate_cols:
    X_train[col], uni = pd.factorize(X_train[col])
y_train = X_train["y"]

best_params = {'bagging_fraction': 0.9, 'bagging_freq': 6, 'feature_fraction': 0.1, 'max_depth': 7, 'min_child_samples': 343, 'min_child_weight': 0.04084861948055769, 'num_leaves': 95, 'reg_alpha': 0.5612212694825488, 'reg_lambda': 0.0001757886119766502}
clf = lgb.LGBMClassifier(n_jobs=-1, seed=71, **best_params)  # 欠損ある場合はGBM使う（リッジより遅い）

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

# + code_folding=[0, 28, 55]
def count_encoder(train_df, valid_df, cat_features=None):
    """
    Count_Encoding: カテゴリ列をカウント値に変換する特徴量エンジニアリング（要はgroupby().size()の集計列追加のこと）
    ※カウント数が同じカテゴリは同じようなデータ傾向になる可能性がある
    https://www.kaggle.com/matleonard/categorical-encodings
    """
    # conda install -c conda-forge category_encoders
    import category_encoders as ce

    if cat_features is None:
        cat_features = train_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()

    count_enc = ce.CountEncoder(cols=cat_features)

    # trainだけでfitすること(validationやtest含めるとリークする)
    count_enc.fit(train_df[cat_features])
    train_encoded = train_df.join(
        count_enc.transform(train_df[cat_features]).add_suffix("_count")
    )
    valid_encoded = valid_df.join(
        count_enc.transform(valid_df[cat_features]).add_suffix("_count")
    )

    return train_encoded, valid_encoded


def target_encoder(train_df, valid_df, target_col: str, cat_features=None):
    """
    Target_Encoding: カテゴリ列を目的変数の平均値に変換する特徴量エンジニアリング
    https://www.kaggle.com/matleonard/categorical-encodings
    """
    # conda install -c conda-forge category_encoders
    import category_encoders as ce

    if cat_features is None:
        cat_features = train_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()

    target_enc = ce.TargetEncoder(cols=cat_features)

    # trainだけでfitすること(validationやtest含めるとリークする)
    target_enc.fit(train_df[cat_features], train_df[target_col])

    train_encoded = train_df.join(
        target_enc.transform(train_df[cat_features]).add_suffix("_target")
    )
    valid_encoded = valid_df.join(
        target_enc.transform(valid_df[cat_features]).add_suffix("_target")
    )
    return train_encoded, valid_encoded


def catboost_encoder(train_df, valid_df, target_col: str, cat_features=None):
    """
    CatBoost_Encoding: カテゴリ列を目的変数の1行前の行からのみに変換する特徴量エンジニアリング
    CatBoost使ったターゲットエンコーディング
    https://www.kaggle.com/matleonard/categorical-encodings
    """
    # conda install -c conda-forge category_encoders
    import category_encoders as ce

    if cat_features is None:
        cat_features = train_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()

    cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)

    # trainだけでfitすること(validationやtest含めるとリークする)
    cb_enc.fit(train_df[cat_features], train_df[target_col])

    train_encoded = train_df.join(
        cb_enc.transform(train_df[cat_features]).add_suffix("_cb")
    )
    valid_encoded = valid_df.join(
        cb_enc.transform(valid_df[cat_features]).add_suffix("_cb")
    )
    return train_encoded, valid_encoded


# + code_folding=[0, 166, 179]
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
        print(
            "Starting LightGBM. Train shape: {}, test shape: {}".format(
                train_df.shape, test_df.shape
            )
        )
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
        train_probas = {}
        test_probas = {}

        # 目的変数とID列など削除
        del_cols = del_cols.append(target_col) if del_cols is not None else [target_col]
        feats = [f for f in train_df.columns if f not in del_cols]

        for n_fold, (train_idx, valid_idx) in tqdm(
            enumerate(folds.split(train_df[feats], train_df[target_col]))
        ):
            t_fold_df = train_df.iloc[train_idx]
            v_fold_df = train_df.iloc[valid_idx]

            # カウントエンコディング
            t_fold_df, v_fold_df = count_encoder(
                t_fold_df, v_fold_df, cat_features=None
            )
            # ターゲットエンコディング
            t_fold_df, v_fold_df = target_encoder(
                t_fold_df, v_fold_df, target_col=target_col, cat_features=None
            )
            # CatBoostエンコディング
            t_fold_df, v_fold_df = catboost_encoder(
                t_fold_df, v_fold_df, target_col=target_col, cat_features=None
            )
            # ラベルエンコディング
            cate_cols = t_fold_df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()
            for col in cate_cols:
                t_fold_df[col], uni = pd.factorize(t_fold_df[col])
                v_fold_df[col], uni = pd.factorize(v_fold_df[col])
            print(
                "run encoding Train shape: {}, valid shape: {}".format(
                    t_fold_df.shape, v_fold_df.shape
                )
            )

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

            ############################ train fit ############################
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
            oof_preds[valid_idx] = clf.predict_proba(
                valid_x, num_iteration=clf.best_iteration_
            )[:, 1]

            ############################ test pred ############################
            # カウントエンコディング
            tr_df, te_df = count_encoder(train_df, test_df, cat_features=None)
            # ターゲットエンコディング
            tr_df, te_df = target_encoder(
                tr_df, te_df, target_col=target_col, cat_features=None
            )
            # CatBoostエンコディング
            tr_df, te_df = catboost_encoder(
                tr_df, te_df, target_col=target_col, cat_features=None
            )
            # ラベルエンコディング
            cate_cols = tr_df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()
            for col in cate_cols:
                tr_df[col], uni = pd.factorize(tr_df[col])
                te_df[col], uni = pd.factorize(te_df[col])
                
            # testの確信度
            test_probas[f"fold_{str(n_fold + 1)}"] = clf.predict_proba(
                te_df[feats], num_iteration=clf.best_iteration_
            )[:, 1]
            sub_preds += test_probas[f"fold_{str(n_fold + 1)}"] / folds.n_splits

            # 一応trainの確信度も出しておく
            train_probas[f"fold_{str(n_fold + 1)}"] = clf.predict_proba(
                tr_df[feats], num_iteration=clf.best_iteration_
            )[:, 1]

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
            feature_importance_df = pd.concat(
                [feature_importance_df, fold_importance_df], axis=0
            )

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
        result_scores_df = pd.DataFrame(
            result_scores.values(), index=result_scores.keys()
        )
        result_scores_df.to_csv(f"{self.OUTPUT_DIR}/result_scores.tsv", sep="\t")

        test_probas_df = pd.DataFrame(test_probas)
        test_probas_df.to_csv(f"{self.OUTPUT_DIR}/test_probas.tsv", index=False)
        train_probas_df = pd.DataFrame(train_probas)
        train_probas_df.to_csv(f"{self.OUTPUT_DIR}/train_probas.tsv", index=False)

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
    df_all = pd.read_csv(f"{OUT_DATA}/feature_add.csv")
    #cols = pd.read_csv(f"{OUT_DATA}/feature_selections.csv")["feature_selections"].values.tolist()
    #cols.append("y")
    cols = ['A1-levelA2-levelA3-levelA4-level_mul', 'A1-rank_A1-weapon_A1-level_diff_mean_x', 'A1-rank_A1-weapon_A1-level_mean_var_x', 'A1-rank_A1-weapon_A1-level_ratio_range_x', 'A1-rank_A1-weapon_A2-level_beyond1std_x', 'A1-rank_A1-weapon_A2-level_kurt_x', 'A1-rank_A1-weapon_A2-level_sem_x', 'A1-rank_A1-weapon_A2-level_skew_x', 'A1-rank_A1-weapon_A2-level_var_x', 'A1-rank_A1-weapon_A3-level_sem_x', 'A1-rank_A1-weapon_A4-level_hl_ratio_x', 'A1-rank_A1-weapon_A4-level_percentile_75_x', 'A1-rank_A1-weapon_B1-level_hl_ratio_x', 'A1-rank_A1-weapon_B1-level_var_x', 'A1-rank_A1-weapon_B3-level_kurt_x', 'A1-rank_A1-weapon_B3-level_var_x', 'A1-rank_A1-weapon_B4-level_kurt_x', 'A1-rank_A1-weapon_B4-level_zscore_x', 'A1-weapon_A2-level_zscore_x', 'A1-weapon_B1-level_beyond1std_x', 'A1-weapon_B4-level_max_x', 'A2-rank_A2-weapon_A1-level_mad', 'A2-rank_A2-weapon_A2-level_mean', 'A2-rank_A2-weapon_A4-level_kurt', 'A2-rank_A2-weapon_A4-level_skew', 'A2-rank_A2-weapon_B2-level_max', 'A2-rank_A2-weapon_B2-level_skew', 'A2-rank_A2-weapon_B2-level_zscore', 'A2-rank_A2-weapon_B3-level_kurt', 'A2-rank_A2-weapon_B4-level_diff_mean', 'A2-rank_A2-weapon_B4-level_diff_percentile_75-25', 'A2-rank_A2-weapon_B4-level_percentile_25', 'A2-weapon_A3-level_ptp_x', 'A2-weapon_A4-level_diff_percentile_75-25_x', 'A2-weapon_B3-level_percentile_75_x', 'A3-rank_A3-weapon_A1-level_sem', 'A3-rank_A3-weapon_A2-level_hl_ratio', 'A3-rank_A3-weapon_A2-level_kurt', 'A3-rank_A3-weapon_A2-level_ratio_range', 'A3-rank_A3-weapon_A3-level_kurt', 'A3-rank_A3-weapon_A4-level_ratio_range', 'A3-rank_A3-weapon_B1-level_diff_mean', 'A3-rank_A3-weapon_B1-level_zscore', 'A3-rank_A3-weapon_B2-level_hl_ratio', 'A3-rank_A3-weapon_B2-level_mean', 'A3-rank_A3-weapon_B2-level_var', 'A3-rank_A3-weapon_B3-level_ratio_mean', 'A3-rank_A3-weapon_B4-level_kurt', 'A3-rank_A3-weapon_B4-level_percentile_25', 'A3-weapon_A1-level_kurt_x', 'A3-weapon_A2-level_skew_x', 'A3-weapon_A3-level_hl_ratio_x', 'A3-weapon_A3-level_max_x', 'A3-weapon_A4-level_percentile_75_x', 'A3-weapon_B3-level_max_x', 'A4-rank_A4-weapon', 'A4-rank_A4-weapon_A1-level_kurt', 'A4-rank_A4-weapon_A3-level_ratio_range', 'A4-rank_A4-weapon_A4-level_diff_mean', 'A4-rank_A4-weapon_A4-level_diff_percentile_75-25', 'A4-rank_A4-weapon_A4-level_ratio_range', 'A4-rank_A4-weapon_B2-level_ratio_range', 'A4-weapon_A1-level_diff_percentile_75-25_x', 'A4-weapon_A1-level_hl_ratio_x', 'A4-weapon_A4-level_mad_x', 'A4-weapon_A4-level_median_x', 'A4-weapon_A4-level_percentile_75_x', 'A4-weapon_A4-level_var_x', 'A4-weapon_B4-level_mad_x', 'A4-weapon_B4-level_mean_var_x', 'A_ranks_A2-level_kurt', 'A_ranks_weapons_B2-level_percentile_25', 'A_weapons_A1-level_min', 'A_weapons_A1-level_prod', 'A_weapons_A2-level_prod', 'A_weapons_A3-level_prod', 'A_weapons_A3-level_sum', 'A_weapons_B1-level_max', 'A_weapons_B1-level_percentile_25', 'A_weapons_B3-level_min', 'A_weapons_B3-level_percentile_75', 'A_weapons_B3-level_prod', 'A_weapons_B4-level_mean', 'B1-levelB2-levelB3-levelB4-level_plus', 'B1-rank_B1-weapon_A1-level_kurt_x', 'B1-rank_B1-weapon_A1-level_var_x', 'B1-rank_B1-weapon_A2-level_kurt_x', 'B1-rank_B1-weapon_A2-level_ptp_x', 'B1-rank_B1-weapon_A2-level_sem_x', 'B1-rank_B1-weapon_A2-level_var_x', 'B1-rank_B1-weapon_A3-level_skew_x', 'B1-rank_B1-weapon_A4-level_skew_x', 'B1-rank_B1-weapon_B1-level_sem_x', 'B1-rank_B1-weapon_B4-level_kurt_x', 'B1-rank_B1-weapon_B4-level_ptp_x', 'B1-weapon', 'B1-weapon_A2-level_percentile_25_x', 'B1-weapon_A2-level_skew_x', 'B2-rank_B2-weapon_A1-level_diff_percentile_75-25', 'B2-rank_B2-weapon_A1-level_ratio_range', 'B2-rank_B2-weapon_A1-level_sem', 'B2-rank_B2-weapon_A4-level_diff_percentile_75-25', 'B2-rank_B2-weapon_A4-level_ptp', 'B2-rank_B2-weapon_A4-level_sem', 'B2-rank_B2-weapon_B1-level_kurt', 'B2-rank_B2-weapon_B2-level_max', 'B2-rank_B2-weapon_B2-level_sem', 'B2-rank_B2-weapon_B3-level_diff_percentile_75-25', 'B2-rank_B2-weapon_B3-level_max', 'B2-rank_B2-weapon_B3-level_mean', 'B2-weapon_A2-level_diff_mean_x', 'B2-weapon_A3-level_percentile_25_x', 'B2-weapon_B1-level_ptp_x', 'B2-weapon_B2-level_hl_ratio_x', 'B2-weapon_B2-level_mean_x', 'B2-weapon_B3-level_var_x', 'B2-weapon_B4-level_max_x', 'B3-rank_B3-weapon_A1-level_ratio_mean', 'B3-rank_B3-weapon_A1-level_sem', 'B3-rank_B3-weapon_A1-level_skew', 'B3-rank_B3-weapon_A2-level_skew', 'B3-rank_B3-weapon_A2-level_sum', 'B3-rank_B3-weapon_A2-level_var', 'B3-rank_B3-weapon_A2-level_zscore', 'B3-rank_B3-weapon_A3-level_hl_ratio', 'B3-rank_B3-weapon_A3-level_ratio_mean', 'B3-rank_B3-weapon_A4-level_max', 'B3-rank_B3-weapon_A4-level_zscore', 'B3-rank_B3-weapon_B1-level_skew', 'B3-rank_B3-weapon_B1-level_zscore', 'B3-rank_B3-weapon_B2-level_kurt', 'B3-rank_B3-weapon_B3-level_diff_mean', 'B3-rank_B3-weapon_B3-level_skew', 'B3-rank_B3-weapon_B3-level_var', 'B3-rank_B3-weapon_B4-level_diff_percentile_75-25', 'B3-rank_B3-weapon_B4-level_hl_ratio', 'B3-rank_B3-weapon_B4-level_kurt', 'B3-weapon_A4-level_beyond1std_x', 'B3-weapon_B3-level_mad_x', 'B3-weapon_B4-level_beyond1std_x', 'B4-level', 'B4-rank_B4-weapon_A1-level_beyond1std', 'B4-rank_B4-weapon_A2-level_diff_mean', 'B4-rank_B4-weapon_A2-level_hl_ratio', 'B4-rank_B4-weapon_A2-level_mean', 'B4-rank_B4-weapon_A3-level_diff_mean', 'B4-rank_B4-weapon_A3-level_max', 'B4-rank_B4-weapon_A3-level_skew', 'B4-rank_B4-weapon_A3-level_zscore', 'B4-rank_B4-weapon_B1-level_sem', 'B4-rank_B4-weapon_B4-level_mad', 'B4-weapon_A1-level_ratio_range_x', 'B4-weapon_A1-level_sum_x', 'B4-weapon_A2-level_mad_x', 'B4-weapon_A3-level_kurt_x', 'B4-weapon_A3-level_ptp_x', 'B4-weapon_B4-level_max_x', 'B_ranks_A1-level_diff_mean', 'B_ranks_A2-level_mad', 'B_ranks_weapons_A1-level_max', 'B_ranks_weapons_A2-level_max', 'B_ranks_weapons_B2-level_min', 'B_weapons', 'B_weapons_A2-level_sum', 'B_weapons_B1-level_prod', 'B_weapons_B2-level_sum', 'B_weapons_B3-level_percentile_75', 'stage_B3-weapon_freq_entropy', 'y']
    df = df_all[cols]

    is_debug = False
    # is_debug = True
    if is_debug:
        _df = df.head(1000)
        _df = _df.append(df.tail(100))
        df = _df

    best_params = {'bagging_fraction': 0.9, 'bagging_freq': 6, 'feature_fraction': 0.1, 'max_depth': 7, 'min_child_samples': 343, 'min_child_weight': 0.04084861948055769, 'num_leaves': 95, 'reg_alpha': 0.5612212694825488, 'reg_lambda': 0.0001757886119766502}
        
    model = Model(OUT_MODEL)
    lgb_params = dict(
        n_estimators=10000,
        learning_rate=0.01,
        silent=-1,
        verbose=-1,
        # verbose=1,
        importance_type="gain",
        **best_params,
    )
    params = dict(
        lgb_params=lgb_params,
        df=df,
        #num_folds=10,
        num_folds=4,
        target_col="y",
        del_cols=None,
        eval_metric="error",
        #stratified=True,
        stratified=False,
        is_submission=True,
        is_plot_perm_importance=False,
    )
    feat_importance, perm_importance = model.kfold_cv_LGBMClassifier(**params)
# -
# # パラメータチューニングしてモデル作成

# + code_folding=[12, 114]
import os
import numpy as np
import pandas as pd
import gc
from joblib import Parallel, delayed
import lightgbm as lgb
import optuna
from sklearn.feature_selection import RFECV
from sklearn.model_selection import *
from sklearn.metrics import *
import seaborn as sns

def exec_study(X, y, cv_index_kv, n_trials, output_dir, t_args):
    def objective(trial):
        gc.collect()
        # ハイパーパラメータ
        max_depth = trial.suggest_int("max_depth", 1, 8)
        num_leaves = trial.suggest_int("num_leaves", 2, 2**max_depth)
        min_child_samples = trial.suggest_int(
            "min_child_samples", 1, max(1, int(len(cv_index_kv[1][0]) / num_leaves)))
        tuning_params = dict(
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            min_child_weight=trial.suggest_loguniform('min_child_weight', 0.001, 1000),
            feature_fraction=trial.suggest_discrete_uniform('feature_fraction', 0.1, 0.95, 0.05),
            bagging_fraction=trial.suggest_discrete_uniform('bagging_fraction', 0.4, 0.95, 0.05),
            bagging_freq=trial.suggest_int('bagging_freq', 1, 10),
            reg_alpha=trial.suggest_loguniform('reg_alpha', 1e-09, 10.0),
            reg_lambda=trial.suggest_loguniform('reg_lambda', 1e-09, 10.0),
        )
        if t_args['objective'] == "regression":
            tuning_params["reg_sqrt"] = trial.suggest_categorical("reg_sqrt", [True, False])
        print(tuning_params)

        # クロスバリデーション
        def calc_score(train_index, val_index):
            train_df = pd.concat([X, y], axis=1)
            t_fold_df = train_df.iloc[train_index]
            v_fold_df = train_df.iloc[val_index]
            # カウントエンコディング
            t_fold_df, v_fold_df = count_encoder(
                t_fold_df, v_fold_df, cat_features=None
            )
            # ターゲットエンコディング
            t_fold_df, v_fold_df = target_encoder(
                t_fold_df, v_fold_df, target_col=target_col, cat_features=None
            )
            # CatBoostエンコディング
            t_fold_df, v_fold_df = catboost_encoder(
                t_fold_df, v_fold_df, target_col=target_col, cat_features=None
            )
            # ラベルエンコディング
            cate_cols = t_fold_df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()
            for col in cate_cols:
                t_fold_df[col], uni = pd.factorize(t_fold_df[col])
                v_fold_df[col], uni = pd.factorize(v_fold_df[col])
            print(
                "run encoding Train shape: {}, valid shape: {}".format(
                    t_fold_df.shape, v_fold_df.shape
                )
            )
            feats = t_fold_df.columns.to_list()
            feats.remove(target_col)
            X_train, y_train = (
                t_fold_df[feats],
                t_fold_df[target_col],
            )
            X_val, y_val = (
                v_fold_df[feats],
                v_fold_df[target_col],
            )
            
            #X_train = X.iloc[train_index]  # TODO df,using_cols,gkfはグローバル変数にしないといけない？
            #y_train = y.iloc[train_index]
            #X_val   = X.iloc[val_index]
            #y_val   = y.iloc[val_index]
            if t_args['objective'] == "regression":
                #model = lgb.LGBMRegressor(n_jobs=1, seed=71, n_estimators=10000, learning_rate=0.1, verbose=-1, **tuning_params)
                model = lgb.LGBMRegressor(n_jobs=-1, seed=71, n_estimators=10000, learning_rate=0.1, verbose=-1, **tuning_params)
            else:
                #model = lgb.LGBMClassifier(n_jobs=1, seed=71, n_estimators=10000, learning_rate=0.1, verbose=-1, **tuning_params)
                model = lgb.LGBMClassifier(n_jobs=-1, seed=71, n_estimators=10000, learning_rate=0.1, verbose=-1, **tuning_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, eval_metric=t_args['eval_metric'], verbose=False)
            
            if t_args["objective"] == "regression":
                score = mean_squared_error(y_val, model.predict(X_val))
            else:
                #score = 1.0 - roc_auc_score(y_val, model.predict(X_val))
                score = 1.0 - accuracy_score(y_val, model.predict(X_val))
            return score
        
        #scores = Parallel(n_jobs=-1)([delayed(calc_score)(train_index, valid_index) for train_index, valid_index in cv_index_kv])
        scores = []
        for train_index, valid_index in cv_index_kv:
            scores = calc_score(train_index, valid_index)
        return np.mean(scores)

    # 学習実行
    study = optuna.create_study(study_name="study",
                                storage=f"sqlite:///{output_dir}/study.db",
                                load_if_exists=True,
                                direction="minimize", 
                                sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
    
    # 学習履歴保存
    study.trials_dataframe().to_csv(f"{output_dir}/study_history.csv", index=False)
    
    # 最適化されたハイパーパラメータ
    return study.best_params.copy()

if __name__ == '__main__':  
    df_all = pd.read_csv(f"{OUT_DATA}/feature_add.csv")
    cols = pd.read_csv(f"{OUT_DATA}/feature_selections.csv")["feature_selections"].values.tolist()
    target_col = "y"
    cols.append(target_col)
    df = df_all[cols]
    #df = df_all   
    print(df.shape)
    
    train = df[df[target_col].notnull()].reset_index(drop=True)
    test = df[df[target_col].isnull()].reset_index(drop=True)
    y_col = "y"
    using_cols = df.columns.to_list()
    using_cols = list(set(using_cols) - set([y_col]))
    
    n_trials = 300
    
    output_dir = OUT_MODEL
    os.makedirs(output_dir, exist_ok=True)
    
    t_args = dict(objective="binary",
                  eval_metric="binary_logloss",
                  #objective="regression",
                  #eval_metric="rmse",
                 )
    
    X_train = df.loc[train.index][using_cols]
    y_train = df.loc[train.index][y_col]
    n_fold = 4
    cv_index_kv = list(KFold(n_fold).split(X_train, y_train))
    #cv_index_kv = list(StratifiedKFold(n_fold).split(X_train, y_train))
    
    # 学習実行
    best_params = exec_study(X_train, y_train, cv_index_kv, n_trials, output_dir, t_args)
    print("best_params:\n", best_params)
    best_params_df = pd.DataFrame(best_params.values(), index=best_params.keys()) 
    best_params_df.to_csv(f"{output_dir}/best_params.tsv", sep="\t")
# -





# + code_folding=[6, 250]
import sys

sys.path.append(r"C:\Users\81908\Git\OptGBM")
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
        print(
            "Starting LightGBM. Train shape: {}, test shape: {}".format(
                train_df.shape, test_df.shape
            )
        )
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
        train_probas = {}
        test_probas = {}

        # 目的変数とID列など削除
        del_cols = del_cols.append(target_col) if del_cols is not None else [target_col]
        feats = [f for f in train_df.columns if f not in del_cols]

        for n_fold, (train_idx, valid_idx) in tqdm(
            enumerate(folds.split(train_df[feats], train_df[target_col]))
        ):
            t_fold_df = train_df.iloc[train_idx]
            v_fold_df = train_df.iloc[valid_idx]

            # カウントエンコディング
            t_fold_df, v_fold_df = count_encoder(
                t_fold_df, v_fold_df, cat_features=None
            )
            # ターゲットエンコディング
            t_fold_df, v_fold_df = target_encoder(
                t_fold_df, v_fold_df, target_col=target_col, cat_features=None
            )
            # CatBoostエンコディング
            t_fold_df, v_fold_df = catboost_encoder(
                t_fold_df, v_fold_df, target_col=target_col, cat_features=None
            )
            # ラベルエンコディング
            cate_cols = t_fold_df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()
            for col in cate_cols:
                t_fold_df[col], uni = pd.factorize(t_fold_df[col])
                v_fold_df[col], uni = pd.factorize(v_fold_df[col])
            print(
                "run encoding Train shape: {}, valid shape: {}".format(
                    t_fold_df.shape, v_fold_df.shape
                )
            )

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

            ############################ train fit ############################
            # LightGBM parameters found by Bayesian optimization
            clf = lgb.LGBMClassifier(**lgb_params)

            clf.fit(
                train_x,
                train_y,
                # eval_set=[(train_x, train_y), (valid_x, valid_y)],
                # eval_metric=eval_metric,
                # verbose=200,
                # early_stopping_rounds=200,
            )

            # モデル保存
            joblib.dump(clf, f"{self.OUTPUT_DIR}/lgb-{n_fold + 1}.model", compress=True)

            # valid pred
            oof_preds[valid_idx] = clf.predict_proba(
                valid_x, num_iteration=clf.best_iteration_
            )[:, 1]

            ############################ test pred ############################
            # カウントエンコディング
            tr_df, te_df = count_encoder(train_df, test_df, cat_features=None)
            # ターゲットエンコディング
            tr_df, te_df = target_encoder(
                tr_df, te_df, target_col=target_col, cat_features=None
            )
            # CatBoostエンコディング
            tr_df, te_df = catboost_encoder(
                tr_df, te_df, target_col=target_col, cat_features=None
            )
            # ラベルエンコディング
            cate_cols = tr_df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()
            for col in cate_cols:
                tr_df[col], uni = pd.factorize(tr_df[col])
                te_df[col], uni = pd.factorize(te_df[col])

            # testの確信度
            test_probas[f"fold_{str(n_fold + 1)}"] = clf.predict_proba(
                te_df[feats], num_iteration=clf.best_iteration_
            )[:, 1]
            sub_preds += test_probas[f"fold_{str(n_fold + 1)}"] / folds.n_splits

            # 一応trainの確信度も出しておく
            train_probas[f"fold_{str(n_fold + 1)}"] = clf.predict_proba(
                tr_df[feats], num_iteration=clf.best_iteration_
            )[:, 1]

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
            feature_importance_df = pd.concat(
                [feature_importance_df, fold_importance_df], axis=0
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
        result_scores_df = pd.DataFrame(
            result_scores.values(), index=result_scores.keys()
        )
        result_scores_df.to_csv(f"{self.OUTPUT_DIR}/result_scores.tsv", sep="\t")

        test_probas_df = pd.DataFrame(test_probas)
        test_probas_df.to_csv(f"{self.OUTPUT_DIR}/test_probas.tsv", index=False)
        train_probas_df = pd.DataFrame(train_probas)
        train_probas_df.to_csv(f"{self.OUTPUT_DIR}/train_probas.tsv", index=False)
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

    # df_all = pd.read_csv(f"{OUT_DATA}/feature_add.csv")
    # cols = pd.read_csv(f"{OUT_DATA}/feature_selections.csv")["feature_selections"].values.tolist()
    # cols.append("y")
    # df = df_all[cols]

    is_debug = False
    # is_debug = True
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





