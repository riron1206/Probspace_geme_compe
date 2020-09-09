"""
特徴量エンジニアリング
Usage:
    $ conda activate tfgpu
    $ python ./feature_engineering.py -o ../data/code_feature_eng
"""
import os
import gc
import sys
import joblib
import argparse
import warnings
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.feature_selection import RFECV, RFE
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
warnings.filterwarnings("ignore")

ORIG = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\orig"


def count_AB_col(df, A_cols, B_cols):
    """A1-4とB1-4単位で指定列の要素ごとのカウント数を集計した列追加。列は要素ごとに作られるので列数大きくなる"""
    df_A = df[A_cols].values
    df_B = df[B_cols].values
    A_count = pd.DataFrame([Counter(x) for x in df_A]).fillna(0).astype("int8")
    B_count = pd.DataFrame([Counter(x) for x in df_B]).fillna(0).astype("int8")
    A_count = A_count.add_suffix("_A")
    B_count = B_count.add_suffix("_B")
    X = pd.concat([A_count, B_count], axis=1)
    return X


def win_rate(col, df):
    """
    追加の武器データのcol列についてA1-4とB1-4単位で集計した列をもとに勝率計算
    列名はcount_AB_col()で付けたものに合わせる
    """
    # それぞれのチームで対象カテゴリーが出現する試合数のカウント
    count = len(df[df[col + "_A"] != 0]) + len(df[df[col + "_B"] != 0])
    # それぞれのチームで対象カテゴリーが出現する試合のうち各チームが勝利する試合数のカウント
    win = len(df[(df[col + "_A"] != 0) & (df.y == 1)]) + len(
        (df[(df[col + "_B"] != 0) & (df.y == 0)])
    )
    rate = win / count
    return rate, count


def add_cate(df, cat_features):
    """文字列を2列結合"""
    for col1, col2 in itertools.combinations(cat_features, 2):
        new_col_name = "_".join([col1, col2])
        new_values = df[col1].map(str) + "_" + df[col2].map(str)
        df[new_col_name] = new_values
    return df


def add_cates(df, cols, new_col_name):
    """複数列一気に連結"""
    df[cols] = df[cols].astype(str)
    df[new_col_name] = df[cols[0]].str.cat(df[cols[1:]], sep="_")
    return df


def concat_combination_xfeat(df):
    """xfeatで列連結"""
    # カテゴリ型の列をobject型に変換（これしないとConcatCombination()エラーになる）
    df = df.apply(lambda x: x.astype(str) if x.dtype.name == "category" else x)
    # 文字列組み合わせ
    df_cate = Pipeline(
        [
            SelectCategorical(  # カテゴリ列のみ選択
                # exclude_cols=["alive"]  # 除外する列
            ),
            ConcatCombination(  # カテゴリ列同士組み合わせ
                drop_origin=True,  # 元の列削除。残す場合はFalse
                r=2,  # 2列単位で組み合わせ。4つの項目から2つを選ぶ場合は　4C2 = 6　6通りが出力
                output_suffix="",  # 列名のサフィックス指定
                fillna=np.nan,  # 結合した値が欠損のときに入れる値。デフォルトは"_NAN_"が入る
            ),
            # LabelEncoder(output_suffix=""),  # ラベルエンコディング
        ]
    ).fit_transform(df)
    return df_cate


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


def add_cate_row_agg(df, agg_cate_cols):
    """行単位の統計量列追加
    agg_cate_cols は文字列でないとエラー"""
    from scipy.stats import entropy

    df = df[agg_cate_cols]
    cols = df.columns.to_list()
    cols = map(str, cols)  # 文字列にする
    col_name = "_".join(cols)

    df[f"{col_name}_freq_entropy"] = df.apply(
        lambda x: entropy(x.value_counts().values), axis=1
    )  # 出現頻度のエントロピー

    def _freq1name(x):
        x = x.dropna()
        return (
            np.nan
            if x.shape[0] == 0
            else x.value_counts().sort_values(ascending=False)[0]
        )

    df[f"{col_name}_freq1name"] = df.apply(_freq1name, axis=1)  # 最も頻繁に出現するカテゴリの数

    def _freq1ratio(x):
        x = x.dropna()
        frq = x.value_counts().sort_values(ascending=False)
        return np.nan if frq.shape[0] == 0 else frq[0] / frq.shape[0]

    df[f"{col_name}_freq1ratio"] = df.apply(_freq1ratio, axis=1)  # 最も頻繁に出現するカテゴリ/グループの数

    return df


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


class FeatureEng:
    @staticmethod
    def time_cols(df, date_col="period"):
        """時刻列ばらしておく"""
        df[date_col] = pd.to_datetime(df[date_col])  # dtype を datetime64 に変換
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["dayofyear"] = df[date_col].dt.dayofyear
        df["dayofweek"] = df[date_col].dt.dayofweek
        df["weekend"] = (df[date_col].dt.dayofweek.values >= 5).astype(int)
        df["hour"] = df[date_col].dt.hour
        return df

    @staticmethod
    def merage_df_add(df_all, buki_csv=f"{ORIG}/statink-weapon2.csv"):
        def _merage_df_add(df_all, df_add, pre):
            """武器情報の追加ファイルを結合"""
            df_add = df_add[
                [
                    "key",
                    "category1",
                    "category2",
                    "mainweapon",
                    "subweapon",
                    "special",
                    "splatnet",
                ]
            ]
            df_add.columns = [pre + col for col in df_add.columns]
            return pd.merge(
                df_all, df_add, how="left", left_on=f"{pre}weapon", right_on=f"{pre}key"
            ).drop(columns=[f"{pre}key"])

        df_add = pd.read_csv(buki_csv)

        # カテゴリ名称とブキ名称が被るためカテゴリ名称を変更する
        # https://prob.space/competitions/game_winner/discussions/uratatsu-Post2240d51da94b313ed72c
        # df_add.loc[df_add["category2"] == "maneuver", "category2"] = "maneuver_cat"

        for pre in ["A1-", "A2-", "A3-", "A4-", "B1-", "B2-", "B3-", "B4-"]:
            df_all = _merage_df_add(df_all, df_add, pre)
        return df_all

    @staticmethod
    def add_null_flag(df):
        """
        参加人数列を追加する
        武器とレベルが欠損はその枠に参加者なしってことっぽい
        武器とレベルが欠損でなくrank(=ウデマエ)欠損はレギュラーマッチのときに起こる。レギューラーマッチはrank関係ないので
        """

        def _add_null_flag(row):
            row["A_n_player"] = row[
                ["A1-weapon", "A2-weapon", "A3-weapon", "A4-weapon"]
            ].count()
            row["B_n_player"] = row[
                ["B1-weapon", "B2-weapon", "B3-weapon", "B4-weapon"]
            ].count()
            return row

        return df.apply(_add_null_flag, axis=1)

    @staticmethod
    def add_num_rank(df):
        """
        rankは順位があるみたいなので順位数にする
        https://wiki.denfaminicogamer.jp/Splatoon2/%E3%82%A6%E3%83%87%E3%83%9E%E3%82%A8
        ウデマエとは、ガチマッチでの、自分の腕前を表す「レート」のようなもの
        ウデマエの段階。ウデマエは、一番低い「C-」からスタート
        C-→C→C+→B-→B→B+→A-→A→A+→S→S+→X
        S+0→S+1→S+2→S+3→→→S+9 2018年4月下旬のアップデートによって、S+10以上はウデマエXとなりました
        """
        dic_rank = {
            "x": 12,
            "s+": 11,
            "s": 10,
            "a+": 9,
            "a": 8,
            "a-": 7,
            "b+": 6,
            "b": 5,
            "b-": 4,
            "c+": 3,
            "c": 2,
            "c-": 1,
        }
        for s in ["A", "B"]:
            for i in [1, 2, 3, 4]:
                df[f"{s}{i}-rank"] = df[f"{s}{i}-rank"].replace(dic_rank)
                df[f"{s}{i}-rank"] = df[f"{s}{i}-rank"].astype(float)
        return df

    @staticmethod
    def add_win_rate(rate_col, df_all, train):
        """A1-4,B1-4の各rate_col列について勝率列追加する"""
        # rate_col = "weapon"
        # rate_col = "rank"
        if rate_col == "rank":
            # ラベルエンコディング済み対策
            rank_cols = [
                "A1-rank",
                "A2-rank",
                "A3-rank",
                "A4-rank",
                "B1-rank",
                "B2-rank",
                "B3-rank",
                "B4-rank",
            ]
            df_all[rank_cols] = df_all[rank_cols].astype(str)
            train[rank_cols] = train[rank_cols].astype(str)

        def _win_rate_df(train_count_AB_col):
            """
            A1-4とB1-4単位でのweapon列の勝率のデータフレーム返す
            Args:
                train_add_rate_col_count: A1-4とB1-4単位で集計した列追加したtrain
            """
            l_win_rate = []
            for col in train_count_AB_col[f"A1-{rate_col}"].unique():
                # 勝率計算
                rate, count = win_rate(col, train_count_AB_col)
                l_win_rate.append([col, rate, count])

            # 勝率をデータフレームに変換
            df_win = pd.DataFrame(l_win_rate, columns=[rate_col, "win_rate", "count"])
            df_win = df_win.sort_values(by="win_rate", ascending=False)
            df_win = df_win.reset_index(drop=True)
            return df_win

        # A1-4とB1-4単位で指定列の要素ごとのカウント数を集計した列追加
        X = count_AB_col(
            train,
            [f"A1-{rate_col}", f"A2-{rate_col}", f"A3-{rate_col}", f"A4-{rate_col}"],
            [f"B1-{rate_col}", f"B2-{rate_col}", f"B3-{rate_col}", f"B4-{rate_col}"],
        )
        train = pd.concat([train, X], axis=1)
        # display(train)

        # 勝率のデータフレーム取得
        df_win_rate = _win_rate_df(train)
        # display(df_win_rate)

        # A1-4,B1-4と紐づける
        for pre in ["A1-", "A2-", "A3-", "A4-", "B1-", "B2-", "B3-", "B4-"]:
            df_all = pd.merge(
                df_all,
                df_win_rate.rename(columns={"win_rate": f"{pre}{rate_col}_win_rate"}),
                how="left",
                left_on=f"{pre}{rate_col}",
                right_on=rate_col,
            ).drop([rate_col, "count"], axis=1)
        return df_all

    @staticmethod
    def add_win_rate_buki(
        rate_col, df_all, train, buki_csv=f"{ORIG}/statink-weapon2.csv"
    ):
        """追加の武器データについてA1-4,B1-4の各rate_col列の勝率列追加する"""
        # 追加の武器データ
        buki_data = pd.read_csv(buki_csv)
        # buki_data.loc[buki_data["category2"] == "maneuver", "category2"] = "maneuver_cat"

        def merge_train_buki_data(train):
            """trainの各AB列に追加の武器情報列追加"""
            cols = [
                "A1-weapon",
                "A2-weapon",
                "A3-weapon",
                "A4-weapon",
                "B1-weapon",
                "B2-weapon",
                "B3-weapon",
                "B4-weapon",
            ]
            for col in cols:
                train = (
                    train.merge(
                        buki_data[
                            [
                                "key",
                                "mainweapon",
                                "subweapon",
                                "special",
                                "category1",
                                "category2",
                            ]
                        ],
                        left_on=col,
                        right_on="key",
                        how="left",
                    )
                    .drop(columns="key")
                    .rename(
                        columns={
                            "mainweapon": "mainweapon" + "_" + col,
                            "subweapon": "subweapon" + "_" + col,
                            "special": "special" + "_" + col,
                            "category1": "category1" + "_" + col,
                            "category2": "category2" + "_" + col,
                        }
                    )
                )
            return train

        def _win_rate_buki_df(rate_col, train_count_AB_col):
            """
            追加の武器データについてA1-4とB1-4単位での指定列の勝率のデータフレーム返す
            Args:
                rate_col: 勝率計算したい追加の武器データ列
                train_add_rate_col_count: count_AB_col()でA1-4とB1-4集計した列を追加したtrain
            """
            l_win_rate = []
            # print(buki_data[rate_col].unique())
            for col in buki_data[rate_col].unique():
                # 勝率計算
                rate, count = win_rate(col, train_count_AB_col)
                l_win_rate.append([col, rate, count])

            # 勝率をデータフレームに変換
            df_win = pd.DataFrame(l_win_rate, columns=[rate_col, "win_rate", "count"])
            df_win = df_win.sort_values(by="win_rate", ascending=False)
            df_win = df_win.reset_index(drop=True)

            return df_win

        def _add_col_win_rate(rate_col, df_all, train):
            """A1-4,B1-4の各rate_col列について勝率列追加する"""
            # trainの各AB列に追加の武器情報列追加
            train = merge_train_buki_data(train)
            X = count_AB_col(
                train,
                [
                    f"{rate_col}_A1-weapon",
                    f"{rate_col}_A2-weapon",
                    f"{rate_col}_A3-weapon",
                    f"{rate_col}_A4-weapon",
                ],
                [
                    f"{rate_col}_B1-weapon",
                    f"{rate_col}_B2-weapon",
                    f"{rate_col}_B3-weapon",
                    f"{rate_col}_B4-weapon",
                ],
            )
            train = pd.concat([train, X], axis=1)
            # display(train)

            # 勝率のデータフレーム取得
            df_win_rate = _win_rate_buki_df(rate_col, train)
            # display(df_win_rate)

            # A1-4,B1-4と紐づける
            for pre in ["A1-", "A2-", "A3-", "A4-", "B1-", "B2-", "B3-", "B4-"]:
                df_all = pd.merge(
                    df_all,
                    df_win_rate.rename(
                        columns={"win_rate": f"{pre}{rate_col}_win_rate"}
                    ),
                    how="left",
                    left_on=f"{pre}{rate_col}",
                    right_on=rate_col,
                ).drop([rate_col, "count"], axis=1)
            return df_all

        return _add_col_win_rate(rate_col, df_all, train)

    @staticmethod
    def add_num_row_agg(df_all):
        def _add_num_row_agg(df_all, agg_num_cols):
            """行単位の統計量列追加
            agg_num_cols は数値列だけでないとエラー"""
            import warnings

            warnings.filterwarnings("ignore")

            df = df_all[agg_num_cols]
            cols = df.columns.to_list()
            cols = map(str, cols)  # 文字列にする
            col_name = "_".join(cols)

            df_all[f"{col_name}_sum"] = df.sum(axis=1).replace(0.0, np.nan)
            df_all[f"{col_name}_mean"] = df.mean(axis=1)
            df_all[f"{col_name}_std"] = df.std(axis=1)
            df_all[f"{col_name}_ratio_range"] = df.max(axis=1) / df.min(axis=1)  # 最大/最小
            df_all[f"{col_name}_mean_var"] = df.std(axis=1) / df.mean(axis=1)  # 平均分散
            df_all[f"{col_name}_ptp"] = df.apply(
                lambda x: np.ptp(x), axis=1
            )  # peak to peak: 最大値と最小値との差
            return df_all

        # rank
        a_rank_cols = ["A1-rank", "A2-rank", "A3-rank", "A4-rank"]
        b_rank_cols = ["B1-rank", "B2-rank", "B3-rank", "B4-rank"]
        df_all[a_rank_cols] = df_all[a_rank_cols].astype(float)
        df_all[b_rank_cols] = df_all[b_rank_cols].astype(float)
        df_all = _add_num_row_agg(df_all, a_rank_cols)
        df_all = _add_num_row_agg(df_all, b_rank_cols)

        for name in ["sum", "mean", "std", "ratio_range", "mean_var", "ptp"]:
            df_all[f"diff_AB-rank_{name}"] = (
                df_all[f"A1-rank_A2-rank_A3-rank_A4-rank_{name}"]
                - df_all[f"B1-rank_B2-rank_B3-rank_B4-rank_{name}"]
            )

        # level
        a_level_cols = ["A1-level", "A2-level", "A3-level", "A4-level"]
        b_level_cols = ["B1-level", "B2-level", "B3-level", "B4-level"]
        df_all = _add_num_row_agg(df_all, a_level_cols)
        df_all = _add_num_row_agg(df_all, b_level_cols)

        for name in ["sum", "mean", "std", "ratio_range", "mean_var", "ptp"]:
            df_all[f"diff_AB-level_{name}"] = (
                df_all[f"A1-level_A2-level_A3-level_A4-level_{name}"]
                - df_all[f"B1-level_B2-level_B3-level_B4-level_{name}"]
            )

        # win_rate
        for name in [
            "weapon_win_rate",
            "rank_win_rate",
            "mainweapon_win_rate",
            "category1_win_rate",
            "category2_win_rate",
            "special_win_rate",
        ]:
            a_level_cols = [f"A1-{name}", f"A2-{name}", f"A3-{name}", f"A4-{name}"]
            b_level_cols = [f"B1-{name}", f"B2-{name}", f"B3-{name}", f"B4-{name}"]
            df_all = _add_num_row_agg(df_all, a_level_cols)
            df_all = _add_num_row_agg(df_all, b_level_cols)

            for stat in ["sum", "mean", "std", "ratio_range", "mean_var", "ptp"]:
                df_all[f"diff_AB-{name}_{stat}"] = (
                    df_all[f"A1-{name}_A2-{name}_A3-{name}_A4-{name}_{stat}"]
                    - df_all[f"B1-{name}_B2-{name}_B3-{name}_B4-{name}_{stat}"]
                )

        return df_all

    @staticmethod
    def AB_groupby(df_all):
        """ABのカテゴリ値列をキーにして数値列集計"""
        for pre in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]:
            num_agg = {}
            for col in [f"{pre}-rank", f"{pre}-level", f"{pre}-splatnet"]:
                num_agg[col] = ["median"]
            for key in [
                f"{pre}-weapon",
                f"{pre}-mainweapon",
                f"{pre}-subweapon",
            ]:
                feature_df = grouping(df_all, key, num_agg, prefix=key + "_")
                df_all = pd.merge(df_all, feature_df, how="left", on=key)

        # カテゴリ値列をキーにしてカテゴリ列集計
        for pre in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]:
            aggs = [
                "count",  # カウント数
                AggUtil().freq_entropy(),  # 出現頻度のエントロピー
                AggUtil().freq1name(),  # 最も頻繁に出現するカテゴリの数
                AggUtil().freq1ratio(),  # 最も頻繁に出現するカテゴリ/グループの数
            ]
            cate_agg = {}
            for col in [f"{pre}-weapon"]:
                cate_agg[col] = aggs
            for key in [
                f"{pre}-rank",
                f"{pre}-category1",
                f"{pre}-category2",
            ]:
                feature_df = grouping(df_all, key, cate_agg, prefix=key + "_")
                df_all = pd.merge(df_all, feature_df, how="left", on=key)

        return df_all

    @staticmethod
    def concat_AB_cates(df_all):
        """Aチーム 4人 vs Bチーム 4人で戦ってるっぽいからA毎、B毎に組み合わせる"""
        a_weapons = [
            "A1-weapon",
            "A2-weapon",
            "A3-weapon",
            "A4-weapon",
        ]
        b_weapons = [
            "B1-weapon",
            "B2-weapon",
            "B3-weapon",
            "B4-weapon",
        ]
        df_all = add_cates(df_all, a_weapons, "A_weapons")
        df_all = add_cates(df_all, b_weapons, "B_weapons")

        df_all = add_cates(
            df_all, ["stage", "mode", "lobby-mode",], "stage_mode_lobby-mode"
        )
        df_all = add_cate(df_all, [f"A_weapons", f"B_weapons"])

        for col in ["stage", "mode", "lobby-mode"]:
            df_all = add_cate(df_all, [f"A_weapons", col])
            df_all = add_cate(df_all, [f"B_weapons", col])
            df_all = add_cate(df_all, [f"A_weapons_B_weapons", col])

        return df_all

    @staticmethod
    def concat_comb_weapon(df_all):
        """weapon列組み合わせる"""
        _weapon_cols = [
            "A1-weapon",
            "A2-weapon",
            "A3-weapon",
            "A4-weapon",
            "B1-weapon",
            "B2-weapon",
            "B3-weapon",
            "B4-weapon",
            "A1-mainweapon",
            "A1-subweapon",
            "A2-mainweapon",
            "A2-subweapon",
            "A3-mainweapon",
            "A3-subweapon",
            "A4-mainweapon",
            "A4-subweapon",
            "B1-mainweapon",
            "B1-subweapon",
            "B2-mainweapon",
            "B2-subweapon",
            "B3-mainweapon",
            "B3-subweapon",
            "B4-mainweapon",
            "B4-subweapon",
        ]
        df_cate_comb = concat_combination_xfeat(df_all[_weapon_cols])
        return pd.concat([df_all, df_cate_comb], axis=1)

    @staticmethod
    def AB_arithmetic(df_all):
        """ABの数値列組み合わせ"""
        # Aチーム 4人 vs Bチーム 4人で戦ってるっぽいからA毎、B毎に組み合わせる
        a_level_cols = ["A1-level", "A2-level", "A3-level", "A4-level"]
        b_level_cols = ["B1-level", "B2-level", "B3-level", "B4-level"]
        a_rank_cols = ["A1-rank", "A2-rank", "A3-rank", "A4-rank"]
        b_rank_cols = ["B1-rank", "B2-rank", "B3-rank", "B4-rank"]

        # levelとrank掛け算
        for l, r in zip(a_level_cols, a_rank_cols):
            try:
                df_all = arithmetic_num_xfeat(
                    df_all, operator="*", n_order=2, num_cols=[l, r]
                )
            except Exception:
                print("ERROR: arithmetic_num_xfeat:", l, r)
        for l, r in zip(b_level_cols, b_rank_cols):
            try:
                df_all = arithmetic_num_xfeat(
                    df_all, operator="*", n_order=2, num_cols=[l, r]
                )
            except Exception:
                print("ERROR: arithmetic_num_xfeat:", l, r)

        # level全体
        for num_cols in [a_level_cols, b_level_cols]:
            df_all = arithmetic_num_xfeat(
                df_all, operator="*", n_order=4, num_cols=num_cols
            )
            df_all = arithmetic_num_xfeat(
                df_all, operator="-", n_order=4, num_cols=num_cols
            )

        # for _suff in ["_sum", "_mean", "_std", "_ratio_range", "_skew", "_kurt", "_ratio_percentile_75-25", "_ptp", "_hl_ratio", "_beyond1std"]:
        for _suff in ["_sum", "_mean", "_std", "_ratio_range", "_ptp"]:
            df_all = arithmetic_num_xfeat(
                df_all,
                operator="*",
                n_order=2,
                num_cols=[
                    f"A1-level_A2-level_A3-level_A4-level{_suff}",
                    f"B1-level_B2-level_B3-level_B4-level{_suff}",
                ],
            )
            df_all = arithmetic_num_xfeat(
                df_all,
                operator="-",
                n_order=2,
                num_cols=[
                    f"A1-level_A2-level_A3-level_A4-level{_suff}",
                    f"B1-level_B2-level_B3-level_B4-level{_suff}",
                ],
            )

        return df_all


def preprocess(train_df, test_df, out_dir):
    """前処理"""
    # 重複レコード削除
    train_df = train_df[~train_df.drop(["id"], axis=1).duplicated()].reset_index(
        drop=True
    )
    df_all = train_df.append(test_df).reset_index(drop=True)

    df_all = FeatureEng().time_cols(df_all)

    # 値がすべて同じ game-ver, lobby列は消しとく。periodやidも多分使わないから消しとく
    df_all = df_all.drop(["id", "period", "game-ver", "lobby"], axis=1)

    df_all.to_csv(f"{out_dir}/preprocess.csv")
    print("INFO: save csv:", f"{out_dir}/preprocess.csv")
    return df_all, train_df


def eng1(out_dir):
    """特徴量エンジニアリング1"""
    df_all = pd.read_csv(f"{out_dir}/preprocess.csv", index_col=0)
    df_all = FeatureEng().merage_df_add(df_all)
    df_all = FeatureEng().add_null_flag(df_all)
    df_all = FeatureEng().add_num_rank(df_all)
    df_all = FeatureEng().add_num_row_agg(df_all)
    df_all.to_csv(f"{out_dir}/eng1.csv")
    print("INFO: save csv:", f"{out_dir}/eng1.csv")
    return df_all


def eng1_1(train_df, test_df, out_dir):
    """特徴量エンジニアリング1_1"""
    df_all, train_df = preprocess(train_df, test_df, out_dir)
    df_all = FeatureEng().merage_df_add(df_all)
    df_all = FeatureEng().add_null_flag(df_all)
    df_all = FeatureEng().add_num_rank(df_all)
    train_df = FeatureEng().add_num_rank(train_df)
    # 勝率追加
    df_all = FeatureEng().add_win_rate("weapon", df_all, train_df)
    df_all = FeatureEng().add_win_rate("rank", df_all, train_df)
    df_all = FeatureEng().add_win_rate_buki("mainweapon", df_all, train_df)
    df_all = FeatureEng().add_win_rate_buki("category1", df_all, train_df)
    df_all = FeatureEng().add_win_rate_buki("category2", df_all, train_df)
    df_all = FeatureEng().add_win_rate_buki("special", df_all, train_df)
    # 行集計
    df_all = FeatureEng().add_num_row_agg(df_all)
    df_all.to_csv(f"{out_dir}/eng1_1.csv")
    print("INFO: save csv:", f"{out_dir}/eng1_1.csv")
    return df_all


def eng2(out_dir):
    """特徴量エンジニアリング2"""
    # df_all = pd.read_csv(f"{out_dir}/eng1.csv", index_col=0)
    df_all = pd.read_csv(f"{out_dir}/eng1_1.csv", index_col=0)
    df_all = FeatureEng().AB_groupby(df_all)
    df_all.to_csv(f"{out_dir}/eng2.csv")
    print("INFO: save csv:", f"{out_dir}/eng2.csv")
    return df_all


def eng3(out_dir):
    """特徴量エンジニアリング3"""
    df_all = pd.read_csv(f"{out_dir}/eng2.csv", index_col=0)
    df_all = FeatureEng().concat_AB_cates(df_all)
    df_all = FeatureEng().concat_comb_weapon(df_all)
    df_all = FeatureEng().AB_arithmetic(df_all)
    df_all.to_csv(f"{out_dir}/eng3.csv")
    print("INFO: save csv:", f"{out_dir}/eng3.csv")
    return df_all


def test_func():
    """
    テスト駆動開発での関数のテスト関数
    test用関数はpythonパッケージの nose で実行するのがおすすめ($ conda install -c conda-forge nose などでインストール必要)
    →noseは再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行する
    $ cd <このモジュールの場所>
    $ nosetests -v -s <本モジュール>.py  # 再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行。-s付けるとprint()の内容出してくれる
    """
    out_dir = "tmp"
    train_df = pd.read_csv(f"{ORIG}/train_data.csv")
    test_df = pd.read_csv(f"{ORIG}/test_data.csv")

    # df_all, _ = preprocess(train_df, test_df, out_dir)
    # df_all = eng1(out_dir)
    df_all = eng1_1(train_df, test_df, out_dir)
    # df_all = eng2(out_dir)
    # df_all = eng3(out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--OUT_DIR", type=str, default="../data/code_feature_eng")
    args = vars(ap.parse_args())

    out_dir = args["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(f"{ORIG}/train_data.csv")
    test_df = pd.read_csv(f"{ORIG}/test_data.csv")

    df_all = eng1_1(train_df, test_df, out_dir)
    df_all = eng2(out_dir)
    df_all = eng3(out_dir)
