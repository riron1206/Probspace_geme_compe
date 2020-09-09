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
import pathlib
import glob

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
pd.set_option("display.max_columns", 300)

# +
OUT_MODEL = (
    r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\model\train"
)
os.makedirs(OUT_MODEL, exist_ok=True)

DATA_DIR = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\feature_eng"

ORIG = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\orig"
train_df = pd.read_csv(f"{ORIG}/train_data.csv")
test_df = pd.read_csv(f"{ORIG}/test_data.csv")
df_all = train_df.append(test_df).reset_index(drop=True)
display(df_all.head(3))


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


# + code_folding=[0]
class Model:
    def __init__(self, OUTPUT_DIR, dict_enc_flag={"count": True, "target": True, "catboost": True}):
        self.OUTPUT_DIR = OUTPUT_DIR
        self.dict_enc_flag = dict_enc_flag

    # LightGBM GBDT with KFold or Stratified KFold
    # Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
    def kfold_cv_LGBMClassifier(
        self,
        lgb_params: dict,
        df: pd.DataFrame,
        num_folds: int,
        target_col: str,
        del_cols=None,
        select_cols=None,
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

            if self.dict_enc_flag["count"]:
                # カウントエンコディング
                t_fold_df, v_fold_df = count_encoder(
                    t_fold_df, v_fold_df, cat_features=None
                )
            if self.dict_enc_flag["target"]:
                # ターゲットエンコディング
                t_fold_df, v_fold_df = target_encoder(
                    t_fold_df, v_fold_df, target_col=target_col, cat_features=None
                )
            if self.dict_enc_flag["catboost"]:
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
            # 指定の列あればそれだけにする
            feats = t_fold_df.columns.to_list() if select_cols is None else select_cols
            if target_col in feats:
                feats.remove(target_col)
            print("len(feats):", len(feats))

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
            if self.dict_enc_flag["count"]:
                # カウントエンコディング
                tr_df, te_df = count_encoder(train_df, test_df, cat_features=None)
            else:
                tr_df, te_df = train_df, test_df
            if self.dict_enc_flag["target"]:
                # ターゲットエンコディング
                tr_df, te_df = target_encoder(tr_df, te_df, target_col=target_col, cat_features=None)
            else:
                tr_df, te_df = tr_df, te_df
            if self.dict_enc_flag["catboost"]:
                # CatBoostエンコディング
                tr_df, te_df = catboost_encoder(tr_df, te_df, target_col=target_col, cat_features=None)
            else:
                tr_df, te_df = tr_df, te_df
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


# + code_folding=[0]
def base_train(df, dict_enc_flag={"count": True, "target": True, "catboost": True}, out_dir=OUT_MODEL):
    is_debug = False
    # is_debug = True
    if is_debug:
        _df = df.head(1000)
        _df = _df.append(df.tail(100))
        df = _df

    best_params = {'subsample': 0.9, 'subsample_freq': 6, 'colsample_bytree': 0.1, 'max_depth': 7, 'min_child_samples': 343, 'min_child_weight': 0.04084861948055769, 'num_leaves': 95, 'reg_alpha': 0.5612212694825488, 'reg_lambda': 0.0001757886119766502}
        
    os.makedirs(out_dir, exist_ok=True)
    model = Model(out_dir, dict_enc_flag)
    lgb_params = dict(
        n_estimators=10000,
        learning_rate=0.01,
        silent=-1,
        #verbose=-1,
        # verbose=1,
        importance_type="gain",
        random_state=71,
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
# # preprocess.csv
# ## target encoding系の列あるほうがいいか確認
# - count + catboost が一番良さそう

df_all = pd.read_csv(f"{DATA_DIR}/preprocess.csv", index_col=0)

base_train(df_all, dict_enc_flag={"count": False, "target": False, "catboost": False})

base_train(df_all, dict_enc_flag={"count": True, "target": False, "catboost": False})

base_train(df_all, dict_enc_flag={"count": False, "target": True, "catboost": False})

base_train(df_all, dict_enc_flag={"count": False, "target": False, "catboost": True})

base_train(df_all, dict_enc_flag={"count": True, "target": False, "catboost": True})

base_train(df_all, dict_enc_flag={"count": True, "target": True, "catboost": False})

base_train(df_all, dict_enc_flag={"count": False, "target": True, "catboost": True})

base_train(df_all, dict_enc_flag={"count": True, "target": True, "catboost": True})



# ## lobby-modeでセグメンテーション
# - regularのエラー率よくなる

# + code_folding=[0]
def train_seg(df):
    """lobby-modeでセグメンテーション"""
    # gachi
    df_gac = df[df["lobby-mode"] == "gachi"]
    df_gac = df_gac.drop("lobby-mode", axis=1)
    base_train(df_gac, dict_enc_flag={"count": True, "target": False, "catboost": True}, out_dir=f"{OUT_MODEL}/gachi")

    # regular
    rank_not_in_cols = [s for s in df.columns.to_list() if 'rank' not in s]  # rankを含まない列名
    df = df[rank_not_in_cols]
    df_reg = df[df["lobby-mode"] == "regular"]
    df_reg = df_reg.drop("lobby-mode", axis=1)
    base_train(df_reg, dict_enc_flag={"count": True, "target": False, "catboost": True}, out_dir=f"{OUT_MODEL}/regular")
    
df_all = pd.read_csv(f"{DATA_DIR}/preprocess.csv", index_col=0)
train_seg(df_all)


# + code_folding=[0, 25, 44]
class Predict:
    """modelファイルから予測する"""
    def __init__(self, csv=f"{DATA_DIR}/preprocess.csv", model_dir=OUT_MODEL):
        self.csv = csv
        self.model_dir = model_dir
    
    @staticmethod
    def _enc(tr_df, te_df, dict_enc_flag):
        if dict_enc_flag["count"]:
            # カウントエンコディング
            tr_df, te_df = count_encoder(tr_df, te_df, cat_features=None)
        if dict_enc_flag["target"]:
            # ターゲットエンコディング
            tr_df, te_df = target_encoder(tr_df, te_df, target_col="y", cat_features=None)
        if dict_enc_flag["catboost"]:
            # CatBoostエンコディング
            tr_df, te_df = catboost_encoder(tr_df, te_df, target_col="y", cat_features=None)
        # ラベルエンコディング
        cate_cols = tr_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
        for col in cate_cols:
            tr_df[col], uni = pd.factorize(tr_df[col])
            te_df[col], uni = pd.factorize(te_df[col])
        return tr_df, te_df

    @staticmethod
    def predict(tr_df, te_df, model_paths, dict_enc_flag):
        sub_preds = np.zeros(te_df.shape[0])
        test_probas = {}
        for m_p in model_paths:
            print(m_p)
            clf = joblib.load(m_p)

            #print(tr_df.shape, te_df.shape)
            tr_df, te_df = Predict()._enc(tr_df, te_df, dict_enc_flag)
            if "y" in te_df.columns.to_list():
                te_df = te_df.drop("y", axis=1)
            # print(tr_df.shape, te_df.shape)

            # testの確信度
            test_probas[f"{pathlib.Path(m_p).name}"] = clf.predict_proba(te_df, num_iteration=clf.best_iteration_)[:, 1]
            # print(test_probas[f"{pathlib.Path(m_p).name}"])
            sub_preds += test_probas[f"{pathlib.Path(m_p).name}"] / len(model_paths)
        return sub_preds

    def main(self, l_mode, dict_enc_flag={"count": True, "target": False, "catboost": True}):
        #l_mode = "regular"
        df_all = pd.read_csv(self.csv, index_col=0)

        if l_mode == "regular":
            rank_not_in_cols = [s for s in df_all.columns.to_list() if 'rank' not in s]  # rankを含まない列名
            df_all = df_all[rank_not_in_cols]
        df_all = df_all[df_all["lobby-mode"] == l_mode]
        df_all = df_all.drop("lobby-mode", axis=1)

        tr_df = df_all[df_all["y"].notnull()]
        te_df = df_all[df_all["y"].isnull()]

        model_paths = sorted(glob.glob(f"{self.model_dir}/*.model"))
        sub_preds = Predict().predict(tr_df, te_df, model_paths, dict_enc_flag)
        #print(sub_preds)
        te_df["sub_pred_mean"] = sub_preds
        return te_df
        
te_df_reg = Predict(model_dir=f"{OUT_MODEL}/regular").main("regular")
te_df_gac = Predict(model_dir=f"{OUT_MODEL}/gachi").main("gachi")
te_df = te_df_reg.append(te_df_gac)
display(te_df.sort_index(axis=1, ascending=True))
# -

te_df["sub_pred_mean"].dropna()





# # eng1.csv

df_all = pd.read_csv(f"{DATA_DIR}/eng1.csv", index_col=0)
base_train(df_all, dict_enc_flag={"count": True, "target": True, "catboost": True})

df_all = pd.read_csv(f"{DATA_DIR}/eng1.csv", index_col=0)
base_train(df_all, dict_enc_flag={"count": True, "target": False, "catboost": True})

# old
df_all = pd.read_csv(f"{DATA_DIR}/eng1.csv", index_col=0)
base_train(df_all, dict_enc_flag={"count": True, "target": False, "catboost": True})

# +
# セグメンテーション
df_all = pd.read_csv(f"{DATA_DIR}/eng1.csv", index_col=0)
train_seg(df_all)

te_df_reg = Predict(csv=f"{DATA_DIR}/eng1.csv", model_dir=f"{OUT_MODEL}/regular").main("regular")
te_df_gac = Predict(csv=f"{DATA_DIR}/eng1.csv", model_dir=f"{OUT_MODEL}/gachi").main("gachi")
te_df = te_df_reg.append(te_df_gac)
display(te_df.sort_index(axis=1, ascending=True))

df_sub = pd.DataFrame({"y": [int(x) for x in te_df["sub_pred_mean"].values >= 0.5]}).reset_index()
df_sub.columns = ["id", "y"]
df_sub.to_csv(f"{OUT_MODEL}/submission_segmentation.csv", index=False)
# -





# # eng2.csv

df_all = pd.read_csv(f"{DATA_DIR}/eng2.csv", index_col=0)
base_train(df_all, dict_enc_flag={"count": True, "target": False, "catboost": True})

# old
df_all = pd.read_csv(f"{DATA_DIR}/eng2.csv", index_col=0)
base_train(df_all, dict_enc_flag={"count": True, "target": False, "catboost": True})



# # rfecv.csv

# +
df_all = pd.read_csv(f"{DATA_DIR}/eng2.csv", index_col=0)

rfecv_cols = pd.read_csv(f"{DATA_DIR}/rfecv_select_cols.csv")["select_cols"].values
#rfecv_cols = pd.read_csv(f"{DATA_DIR}/rfecv.csv").columns.to_list()
df_all = df_all[rfecv_cols]

base_train(df_all, dict_enc_flag={"count": True, "target": False, "catboost": True})
# -



# # feature_selection_cols.csv

# + code_folding=[0]
def train_select_cols(df, dict_enc_flag={"count": True, "target": True, "catboost": True}, out_dir=OUT_MODEL):
    is_debug = False
    # is_debug = True
    if is_debug:
        _df = df.head(1000)
        _df = _df.append(df.tail(100))
        df = _df

    best_params = {'subsample': 0.9, 'subsample_freq': 6, 'colsample_bytree': 0.1, 'max_depth': 7, 'min_child_samples': 343, 'min_child_weight': 0.04084861948055769, 'num_leaves': 95, 'reg_alpha': 0.5612212694825488, 'reg_lambda': 0.0001757886119766502}
        
    os.makedirs(out_dir, exist_ok=True)
    model = Model(out_dir, dict_enc_flag)
    lgb_params = dict(
        n_estimators=10000,
        learning_rate=0.01,
        silent=-1,
        #verbose=-1,
        # verbose=1,
        importance_type="gain",
        random_state=71,
        **best_params,
    )
    params = dict(
        lgb_params=lgb_params,
        df=df,
        num_folds=10,
        #num_folds=4,
        target_col="y",
        del_cols=None,
        select_cols=pd.read_csv(f"{DATA_DIR}/feature_selection_cols.csv")["feature_selections"].to_list(),
        eval_metric="error",
        #stratified=True,
        stratified=False,
        is_submission=True,
        is_plot_perm_importance=False,
    )
    feat_importance, perm_importance = model.kfold_cv_LGBMClassifier(**params)
    
df_all = pd.read_csv(f"{DATA_DIR}/eng2.csv", index_col=0)
train_select_cols(df_all)
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
            feature_fraction=trial.suggest_discrete_uniform('colsample_bytree', 0.1, 0.95, 0.05),  # feature_fractionのこと
            bagging_fraction=trial.suggest_discrete_uniform('subsample', 0.4, 0.95, 0.05),  # bagging_fractionのこと
            bagging_freq=trial.suggest_int('subsample_freq', 1, 10),  # bagging_freqのこと
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
    df_all = pd.read_csv(f"{WORK_DIR}/eda_preprocess.csv", index_col=0)
    # df_all = pd.read_csv(f"{OUT_DATA}/feature_add.csv")
    # cols = pd.read_csv(f"{OUT_DATA}/feature_selections.csv")["feature_selections"].values.tolist()
    target_col = "y"
    #cols.append(target_col)
    #df = df_all[cols]
    df = df_all   
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
# + code_folding=[6, 282]
import sys

sys.path.append(r"C:\Users\81908\Git\OptGBM")
import optgbm as lgb


class Model:
    def __init__(self, OUTPUT_DIR, dict_enc_flag={"count": True, "target": True, "catboost": True}):
        self.OUTPUT_DIR = OUTPUT_DIR
        self.dict_enc_flag = dict_enc_flag

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

            if self.dict_enc_flag["count"]:
                # カウントエンコディング
                t_fold_df, v_fold_df = count_encoder(
                    t_fold_df, v_fold_df, cat_features=None
                )
            if self.dict_enc_flag["target"]:
                # ターゲットエンコディング
                t_fold_df, v_fold_df = target_encoder(
                    t_fold_df, v_fold_df, target_col=target_col, cat_features=None
                )
            if self.dict_enc_flag["catboost"]:
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
            if self.dict_enc_flag["count"]:
                # カウントエンコディング
                tr_df, te_df = count_encoder(train_df, test_df, cat_features=None)
            else:
                tr_df, te_df = train_df, test_df
            if self.dict_enc_flag["target"]:
                # ターゲットエンコディング
                tr_df, te_df = target_encoder(tr_df, te_df, target_col=target_col, cat_features=None)
            else:
                tr_df, te_df = tr_df, te_df
            if self.dict_enc_flag["catboost"]:
                # CatBoostエンコディング
                tr_df, te_df = catboost_encoder(tr_df, te_df, target_col=target_col, cat_features=None)
            else:
                tr_df, te_df = tr_df, te_df
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
    df = df_all

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
        random_state=71,
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













