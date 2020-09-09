"""
Usage:
    $ conda activate tfgpu
    $ python ./util.py -p ../model/train
    $ python ./util.py -p ../model/code_train/eng2_csv
    $ python ./util.py -p ../model/code_train/eng2_csv -th 0.5  # 閾値指定する場合
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import *


class Util:
    @staticmethod
    def check_y_diff(
        out_dir,
        train_prob_csv,
        train_csv=r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\orig\train_data.csv",
    ):
        """trainの正解ラベルと予測ラベルの差確認"""
        train_df = pd.read_csv(train_csv)
        train_df = train_df[~train_df.drop(["id"], axis=1).duplicated()].reset_index(
            drop=True
        )  # 重複レコード削除
        train_prob = pd.read_csv(train_prob_csv)
        train_prob["cv_mean"] = train_prob.apply(
            lambda x: np.mean(x), axis=1
        ).values  # cvの平均値
        df_con = pd.concat([train_df, train_prob["cv_mean"]], axis=1)
        df_con["diff"] = np.abs(df_con["y"] - df_con["cv_mean"])
        df_con = df_con.sort_values("diff", ascending=True)
        df_con["diff"].tail(20).plot.barh(figsize=(8, 10))
        plt.savefig(f"{out_dir}/cv_mean_diff.png")
        df_con.to_csv(f"{out_dir}/cv_mean_diff.csv")


class Submit:
    @staticmethod
    def nelder_mead_th(true_y, pred_y):
        """ネルダーミードでf1スコアから2値分類のbestな閾値見つける"""
        from scipy.optimize import minimize

        def opt(x):
            return -accuracy_score(true_y, pred_y >= x)
            return -f1_score(true_y, pred_y >= x)

        result = minimize(opt, x0=np.array([0.5]), method="Nelder-Mead")
        best_threshold = result["x"].item()
        return best_threshold

    @staticmethod
    def pred_nelder_mead(pred_dir, threshold):
        """nelder_meadで決めた閾値でcvの平均値を2値化する"""

        train_df = pd.read_csv("../data/orig/train_data.csv")
        # 重複レコード削除
        train_df = train_df[~train_df.drop(["id"], axis=1).duplicated()]
        train_y = train_df["y"].values

        # nelder_meadの学習データ
        df_pred = pd.read_csv(f"{pred_dir}/train_probas.csv", sep=",")

        # cvの平均値
        train_pred_prob = df_pred.apply(lambda x: np.mean(x), axis=1).values

        # 閾値を0.5としたとき
        init_threshold = 0.5
        init_score = f1_score(train_y, train_pred_prob >= init_threshold)
        print("init_threshold, init_score:", init_threshold, init_score)

        if threshold is None:
            # nelder_meadの閾値
            best_threshold = Submit().nelder_mead_th(train_y, train_pred_prob)
            best_score = f1_score(train_y, train_pred_prob >= best_threshold)
            print("best_threshold, best_score:", best_threshold, best_score)
        else:
            # 閾値固定
            best_threshold = threshold

        # nelder_meadの閾値で2値化
        df_pred = pd.read_csv(f"{pred_dir}/test_probas.csv", sep=",")
        test_pred_prob = df_pred.apply(lambda x: np.mean(x), axis=1).values
        test_pred_y = [int(x > best_threshold) for x in test_pred_prob]

        # ファイル出力
        df_test_pred = pd.DataFrame({"y": test_pred_y}).reset_index()
        df_test_pred.columns = ["id", "y"]
        out_csv = f"{pred_dir}/submission_nelder_mead.csv"
        df_test_pred.to_csv(out_csv, index=False)
        # display(df_test_pred)
        print("INFO: save csv:", out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p", "--pred_dir", type=str,
    )
    ap.add_argument("-th", "--threshold", type=float, default=None, help="閾値固定する場合")
    args = vars(ap.parse_args())
    # nelder_meadで決めた閾値でcvの平均値を2値化して出力
    Submit().pred_nelder_mead(args["pred_dir"], args["threshold"])
