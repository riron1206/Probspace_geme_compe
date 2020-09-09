"""
lgbで学習
Usage:
    $ conda activate tfgpu

    # ymlでパラメータ指定してモデル作成
    # LB=0.553140(submission_kernel.csv) cv_error=0.440300  2020/08/30
    $ python ./train_lgb.py \
        -o ../model/code_train/code_eng1_1_csv/train_best_param \
        -i ../data/code_feature_eng/eng1_1.csv \
        -y config/best_params_eng1_1_csv_tuining.yml

    $ python ./train_lgb.py -o ../model/code_train/code_eng3_remove_useless_features -i ../data/feature_select/code_eng3_csv/remove_useless_features.csv -y config/best_params_eng2_csv_tuining.yml  # cv_error=0.470615 2020/08/29
    $ python ./train_lgb.py -o ../model/code_train/code_eng3_features_select -i ../data/code_feature_eng/eng3.csv -y config/best_params_eng1_1_csv_tuining_select_cols.yml  # cv_error= 2020/08/30

    # パラメータチューニング
    $ python ./train_lgb.py -o ../model/code_train/eng2_csv/tuning -i ../data/feature_eng/eng2.csv -n_t 150
    $ python ./train_lgb.py -o ../model/code_train/code_eng1_1_csv/tuning -i ../data/code_feature_eng/eng1_1.csv -n_t 300

    # 乱数とboosting_type変えて学習してソフトアンサンブル
    $ python ./train_lgb.py -o ../model/code_train/code_eng1_1_csv/tuning -i ../data/code_feature_eng/eng1_1.csv -is_e

"""
import os
import gc
import sys
import joblib
import warnings
import itertools
import pathlib
import glob
import yaml

import argparse
import numpy as np
import pandas as pd
import matplotlib
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
from lightgbm import *
import optuna
from traitlets.traitlets import default

sys.path.append(r"C:\Users\81908\Git\xfeat")
import xfeat
from xfeat import *
from xfeat.selector import *
from xfeat.utils import *

sys.path.append(r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\code")
from util import Util

matplotlib.use("Agg")
sns.set()
warnings.simplefilter(action="ignore", category=FutureWarning)

DATA_DIR = "../data/feature_eng"


class Model:
    def __init__(
        self,
        output_dir=None,
        dict_enc_flag={"count": True, "target": True, "catboost": True},
    ):
        self.output_dir = output_dir
        self.dict_enc_flag = dict_enc_flag

    def _encoding(self, tr_df, te_df, target_col):
        """ターゲットエンコディングとか一括実行"""
        if self.dict_enc_flag["count"]:
            # カウントエンコディング
            tr_df, te_df = Model().count_encoder(tr_df, te_df, cat_features=None)
        if self.dict_enc_flag["target"]:
            # ターゲットエンコディング
            tr_df, te_df = Model().target_encoder(
                tr_df, te_df, target_col=target_col, cat_features=None
            )
        if self.dict_enc_flag["catboost"]:
            # CatBoostエンコディング
            tr_df, te_df = Model().catboost_encoder(
                tr_df, te_df, target_col=target_col, cat_features=None
            )
        # ラベルエンコディング
        cate_cols = tr_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()
        for col in cate_cols:
            tr_df[col], uni = pd.factorize(tr_df[col])
            te_df[col], uni = pd.factorize(te_df[col])

        return tr_df, te_df

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
        random_state=1001,
    ):
        """
        LGBMClassifierでcross validation + feature_importance/permutation importance plot
        """
        # データフレームからID列など不要な列削除
        if del_cols is not None:
            df = df.drop(del_cols, axis=1)

        # 特徴量の列のみ保持
        feats = df.columns.to_list()
        feats.remove(target_col)

        # Divide in training/validation and test data
        train_df = df[df[target_col].notnull()].reset_index(drop=True)
        test_df = df[df[target_col].isnull()].reset_index(drop=True)
        print(
            f"INFO: Starting LightGBM. Train shape: {train_df.shape}, test shape: {test_df.shape}"
        )
        del df
        gc.collect()

        ###################################### cross validation ######################################
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(
                n_splits=num_folds, shuffle=True, random_state=random_state
            )
        else:
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

        # Create arrays and dataframes to store results
        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        feature_importance_df = pd.DataFrame()
        permutation_importance_df = pd.DataFrame()
        result_scores = {}
        train_probas = {}
        test_probas = {}
        best_threshold = 0.0

        for n_fold, (train_idx, valid_idx) in enumerate(
            folds.split(train_df[feats], train_df[target_col])
        ):
            print(
                f"\n------------------------------------ n_fold={n_fold + 1} ------------------------------------"
            )
            ############################ create fold ############################
            fold_df_base = train_df.iloc[train_idx]  # エンコディングのベースデータ
            v_fold_df = train_df.iloc[valid_idx]

            # ターゲットエンコディングとか一括実行
            t_fold_df, v_fold_df = self._encoding(fold_df_base, v_fold_df, target_col)
            print(
                f"INFO: Encoded Train shape: {t_fold_df.shape}, valid shape: {v_fold_df.shape}"
            )

            # 指定の列あればそれだけにする
            feats = t_fold_df.columns.to_list() if select_cols is None else select_cols
            if target_col in feats:
                feats.remove(target_col)
            print(f"INFO: select features: {len(feats)}\n")

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
            joblib.dump(clf, f"{self.output_dir}/lgb-{n_fold + 1}.model", compress=True)
            # モデルのパラメ
            pd.DataFrame.from_dict(clf.get_params(), orient="index").to_csv(
                f"{self.output_dir}/param.tsv", sep="\t",
            )

            ############################ valid pred ############################
            oof_preds[valid_idx] = clf.predict_proba(
                valid_x, num_iteration=clf.best_iteration_
            )[:, 1]
            if eval_metric == "auc":
                fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
                print("\nINFO: Fold %2d AUC : %.6f" % (n_fold + 1, fold_auc))
                result_scores[f"fold_auc_{str(n_fold + 1)}"] = fold_auc
            elif eval_metric == "error":
                # intにしないとaccuracy_score()エラーになる
                _pred = oof_preds[valid_idx]
                _pred[_pred >= 0.5] = 1
                _pred[_pred < 0.5] = 0
                fold_err = 1.0 - accuracy_score(valid_y, _pred)
                print(
                    "\nINFO: Fold %2d error(threshold=0.5) : %.6f"
                    % (n_fold + 1, fold_err)
                )
                result_scores[f"fold_err_{str(n_fold + 1)}"] = fold_err

                # best_thresholdで2値化
                _pred = oof_preds[valid_idx]
                _best_threshold = Model().nelder_mead_th(valid_y, _pred)
                _pred[_pred >= _best_threshold] = 1
                _pred[_pred < _best_threshold] = 0
                fold_err_best_threshold = 1.0 - accuracy_score(valid_y, _pred)
                print(
                    f"\nINFO: Fold %2d error(threshold={_best_threshold}) : %.6f"
                    % (n_fold + 1, fold_err_best_threshold)
                )
                best_threshold += _best_threshold / num_folds

            ############################ test pred ############################
            if test_df.shape[0] > 0:
                # ターゲットエンコディングとか一括実行
                tr_df, te_df = self._encoding(fold_df_base, test_df, target_col)

                # testの確信度
                test_probas[f"fold_{str(n_fold + 1)}"] = clf.predict_proba(
                    te_df[feats], num_iteration=clf.best_iteration_
                )[:, 1]
                sub_preds += test_probas[f"fold_{str(n_fold + 1)}"] / folds.n_splits

                # 一応trainの確信度も出しておく
                train_probas[f"fold_{str(n_fold + 1)}"] = clf.predict_proba(
                    tr_df[feats], num_iteration=clf.best_iteration_
                )[:, 1]

            ############################ importance 計算 ############################
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

        print(
            "\n------------------------------------ mean fold ------------------------------------"
        )
        mean_fold_score = None
        if eval_metric == "auc":
            mean_fold_score = roc_auc_score(train_df[target_col], oof_preds)
            print("INFO: Mean valid AUC score %.6f" % mean_fold_score)
            result_scores["mean_fold_auc"] = mean_fold_score
        elif eval_metric == "error":
            # intにしないとaccuracy_score()エラーになる
            _pred = oof_preds
            _pred[_pred >= 0.5] = 1
            _pred[_pred < 0.5] = 0
            mean_fold_score = 1.0 - accuracy_score(train_df[target_col], _pred)
            print("INFO: Mean valid error score %.6f" % mean_fold_score)
            result_scores["mean_fold_err"] = mean_fold_score

        # モデルの評価指標出力
        result_scores_df = pd.DataFrame(
            result_scores.values(), index=result_scores.keys()
        )
        result_scores_df.to_csv(f"{self.output_dir}/result_scores.tsv", sep="\t")

        # test setについて
        if test_df.shape[0] > 0:
            test_probas_df = pd.DataFrame(test_probas)
            test_probas_df.to_csv(f"{self.output_dir}/test_probas.csv", index=False)
            ############################ Write submission file ############################
            if is_submission:
                # threshold=0.5で2値化
                te_mean = test_probas_df.apply(lambda x: np.mean(x), axis=1).values
                te_mean[te_mean >= 0.5] = 1
                te_mean[te_mean < 0.5] = 0
                te_mean = te_mean.astype(int)
                output_csv = f"{self.output_dir}/submission_kernel.csv"
                pd.DataFrame({"id": range(len(te_mean)), "y": te_mean}).to_csv(
                    output_csv, index=False
                )
                print(f"INFO: save csv {output_csv}")

                # best_thresholdで2値化
                print(f"INFO: submission best_threshold(cv mean): {best_threshold}")
                te_mean = test_probas_df.apply(lambda x: np.mean(x), axis=1).values
                te_mean[te_mean >= best_threshold] = 1
                te_mean[te_mean < best_threshold] = 0
                te_mean = te_mean.astype(int)
                output_csv = f"{self.output_dir}/submission_nelder_mead.csv"
                pd.DataFrame({"id": range(len(te_mean)), "y": te_mean}).to_csv(
                    output_csv, index=False
                )
                print(f"INFO: save csv {output_csv}")

        # Plot feature importance
        png_path = f"{self.output_dir}/lgbm_feature_importances.png"
        Model().display_importances(
            feature_importance_df, png_path=png_path, title="feature_importance",
        )
        # print(f"INFO: save png {png_path}")
        if is_plot_perm_importance:
            png_path = f"{self.output_dir}/lgbm_permutation_importances.png"
            Model().display_importances(
                permutation_importance_df,
                png_path=png_path,
                title="permutation_importance",
            )
            # print(f"INFO: save png {png_path}")

        return mean_fold_score, feature_importance_df, permutation_importance_df

    def exec_study(self, params, n_trials):
        """kfold_cv_LGBMClassifier()使ってパラメータチューニング"""

        def objective(trial):
            gc.collect()
            # ハイパーパラメータ
            max_depth = trial.suggest_int("max_depth", 1, 8)
            num_leaves = trial.suggest_int("num_leaves", 2, 2 ** max_depth)
            min_child_samples = trial.suggest_int(
                "min_child_samples", 1, max(1, params["df"].shape[0] // num_leaves)
            )
            tuning_params = dict(
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                min_child_weight=trial.suggest_loguniform(
                    "min_child_weight", 0.001, 1000
                ),
                feature_fraction=trial.suggest_discrete_uniform(
                    "colsample_bytree", 0.1, 0.95, 0.05
                ),  # feature_fractionのこと
                bagging_fraction=trial.suggest_discrete_uniform(
                    "subsample", 0.4, 0.95, 0.05
                ),  # bagging_fractionのこと
                bagging_freq=trial.suggest_int(
                    "subsample_freq", 1, 10
                ),  # bagging_freqのこと
                reg_alpha=trial.suggest_loguniform("reg_alpha", 1e-09, 10.0),
                reg_lambda=trial.suggest_loguniform("reg_lambda", 1e-09, 10.0),
                # drop_rate = trial.suggest_loguniform('drop_rate', 1e-8, 1.0), # dartのとき用
                # skip_drop = trial.suggest_loguniform('skip_drop', 1e-8, 1.0), # dartのとき用
                # top_rate = trial.suggest_uniform('top_rate', 0.0, 1.0)  # gossのとき用
                # other_rate = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])  # gossのとき用
            )
            print(tuning_params)

            lgb_params = dict(
                n_estimators=10000,
                learning_rate=0.1,
                silent=-1,
                importance_type="gain",
                random_state=71,
                **tuning_params,
            )
            params["lgb_params"] = lgb_params
            mean_fold_score, _, _ = self.kfold_cv_LGBMClassifier(**params)
            return mean_fold_score

        # 学習実行
        study = optuna.create_study(
            study_name="study",
            storage=f"sqlite:///{self.output_dir}/study.db",
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=1),
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
        # 学習履歴保存
        study.trials_dataframe().to_csv(
            f"{self.output_dir}/study_history.csv", index=False
        )
        # 最適化されたハイパーパラメータ
        pd.DataFrame(study.best_params.values(), index=study.best_params.keys()).to_csv(
            f"{self.output_dir}/best_params.tsv", sep="\t"
        )
        print(f"best_params: {study.best_params}")
        return study.best_params

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def catboost_encoder(
        train_df, valid_df, target_col: str, cat_features=None, random_state=7
    ):
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

        cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=random_state)

        # trainだけでfitすること(validationやtest含めるとリークする)
        cb_enc.fit(train_df[cat_features], train_df[target_col])

        train_encoded = train_df.join(
            cb_enc.transform(train_df[cat_features]).add_suffix("_cb")
        )
        valid_encoded = valid_df.join(
            cb_enc.transform(valid_df[cat_features]).add_suffix("_cb")
        )
        return train_encoded, valid_encoded

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
        plt.subplots_adjust(left=0.5)  # 特徴量の名前が長すぎてグラフのラベルがはみ出てしまう対策
        # plt.tight_layout()
        plt.savefig(png_path)

    @staticmethod
    def nelder_mead_th(true_y, pred_y):
        """ネルダーミードでf1スコアから2値分類のbestな閾値見つける"""
        from scipy.optimize import minimize

        def opt(x):
            return -accuracy_score(true_y, pred_y >= x)
            # return -f1_score(true_y, pred_y >= x)

        result = minimize(opt, x0=np.array([0.5]), method="Nelder-Mead")
        best_threshold = result["x"].item()
        return best_threshold


def train_select_cols(
    df,
    out_dir,
    config,
    target_col="y",
    dict_enc_flag={"count": True, "target": False, "catboost": True},
    is_debug=False,
):
    if is_debug:
        _df = df.head(1000)
        _df = _df.append(df.tail(100))
        df = _df

    os.makedirs(out_dir, exist_ok=True)
    model = Model(out_dir, dict_enc_flag)

    lgb_params = dict(
        n_estimators=10000,
        learning_rate=0.01,
        silent=-1,
        importance_type="gain",
        # random_state=71,
        **config["best_params"],
    )
    if "select_cols_csv" in config:
        select_cols = pd.read_csv(config["select_cols_csv"])[
            "feature_selections"
        ].to_list()
    else:
        select_cols = None
    params = dict(
        lgb_params=lgb_params,
        df=df,
        num_folds=10,
        # num_folds=4,
        target_col=target_col,
        del_cols=None,
        select_cols=select_cols,
        eval_metric="error",
        # stratified=True,
        stratified=False,
        is_submission=True,
        is_plot_perm_importance=False,
    )
    mean_fold_score, feat_importance, perm_importance = model.kfold_cv_LGBMClassifier(
        **params
    )


def tuning(
    df,
    out_dir,
    target_col="y",
    n_trials=50,
    dict_enc_flag={"count": True, "target": False, "catboost": True},
    is_debug=False,
):
    if is_debug:
        _df = df.head(1000)
        _df = _df.append(df.tail(100))
        df = _df

    os.makedirs(out_dir, exist_ok=True)
    model = Model(out_dir, dict_enc_flag)

    params = dict(
        df=df,
        num_folds=3,
        target_col=target_col,
        del_cols=None,
        # select_cols=pd.read_csv(f"{DATA_DIR}/feature_selection_cols.csv")[
        #    "feature_selections"
        # ].to_list(),
        eval_metric="error",
        # stratified=True,
        stratified=False,
        is_submission=False,
        is_plot_perm_importance=False,
    )
    best_params = model.exec_study(params, n_trials)


def train_soft_ensemble(
    df,
    out_dir,
    config,
    target_col="y",
    dict_enc_flag={"count": True, "target": False, "catboost": True},
):
    """乱数とboosting_type変えて学習してソフトアンサンブルする"""
    test_pred_probs = {}
    # for _boosting_type in ["gbdt", "dart"]:
    for _boosting_type in ["gbdt"]:
        for i, _random_state in enumerate([3, 71, 847, 1258, 45978, 2365472]):
            best_params = config["best_params"]
            # boosting_type変更
            best_params["boosting_type"] = _boosting_type
            # lgbの乱数変更
            best_params["random_state"] = _random_state
            # set param
            config["best_params"] = best_params
            # 学習
            train_select_cols(
                df, out_dir, config, target_col=target_col, dict_enc_flag=dict_enc_flag
            )
            # 予測結果取得
            df_pred = pd.read_csv(f"{out_dir}/test_probas.csv", sep=",")
            test_pred_probs[f"{_boosting_type}_{i}"] = df_pred.apply(
                lambda x: np.mean(x), axis=1
            ).values
    # ソフトアンサンブル
    train_probas_df = pd.DataFrame(test_pred_probs)
    ensemble_pb = train_probas_df.apply(lambda x: np.mean(x), axis=1).values
    ensemble_y = [int(x >= 0.5) for x in ensemble_pb]
    output_csv = f"{out_dir}/submission_ensemble.csv"
    pd.DataFrame({"id": range(len(train_probas_df)), target_col: ensemble_y}).to_csv(
        output_csv, index=False
    )
    print(f"INFO: save csv {output_csv}")


def test_func():
    """
    テスト駆動開発での関数のテスト関数
    test用関数はpythonパッケージの nose で実行するのがおすすめ($ conda install -c conda-forge nose などでインストール必要)
    →noseは再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行する
    $ cd <このモジュールの場所>
    $ nosetests -v -s <本モジュール>.py  # 再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行。-s付けるとprint()の内容出してくれる
    """

    def _test_train():
        """testデータ(タイタニック)でtrain実行"""
        df = sns.load_dataset("titanic")
        df.iloc[700:, 0] = np.nan  # test set用意
        output_dir = "tmp"
        os.makedirs(output_dir, exist_ok=True)
        lgb_params = dict(silent=-1, random_state=71, importance_type="gain",)
        params = dict(
            lgb_params=lgb_params,
            df=df,
            num_folds=4,
            target_col="survived",
            del_cols=["alive"],
            select_cols=None,
            eval_metric="error",  # "auc",
            stratified=False,  # True
            is_submission=True,  # False
            is_plot_perm_importance=True,
        )
        model = Model(
            output_dir=output_dir,
            dict_enc_flag={"count": True, "target": True, "catboost": True},
            # dict_enc_flag={"count": False, "target": False, "catboost": False},
        )
        mean_fold_score, _, _, = model.kfold_cv_LGBMClassifier(**params)

    def _test_optuna():
        """testデータ(タイタニック)でパラメータチューニング実行"""
        df = sns.load_dataset("titanic")
        df.iloc[700:, 0] = np.nan  # test set用意
        output_dir = "tmp"
        os.makedirs(output_dir, exist_ok=True)
        params = dict(
            df=df,
            num_folds=4,
            target_col="survived",
            del_cols=["alive"],
            select_cols=None,
            eval_metric="error",  # "auc",
            stratified=False,  # True
            is_submission=False,
            is_plot_perm_importance=False,
        )
        model = Model(
            output_dir=output_dir,
            dict_enc_flag={"count": True, "target": True, "catboost": True},
            # dict_enc_flag={"count": False, "target": False, "catboost": False},
        )
        best_params = model.exec_study(params, n_trials=10)

    def _test_train_soft_ensemble():
        """testデータ(タイタニック)でtrain_soft_ensemble()実行"""
        df = sns.load_dataset("titanic")
        df.iloc[700:, 0] = np.nan  # test set用意
        df = df.drop("alive", axis=1)
        output_dir = "tmp"
        os.makedirs(output_dir, exist_ok=True)
        config = {"best_params": dict(random_state=71,)}
        train_soft_ensemble(df, output_dir, config, target_col="survived")

    # _test_train()
    # _test_optuna()
    _test_train_soft_ensemble()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o", "--OUT_MODEL", type=str,
    )
    ap.add_argument(
        "-i", "--INPUT_CSV", type=str,
    )
    ap.add_argument("-y", "--YML", type=str, default="config/best_params1.yml")
    ap.add_argument("-n_t", "--n_trials", type=int, default=0)
    ap.add_argument(
        "-is_e",
        "--is_ensemble",
        action="store_const",
        const=True,
        default=False,
        help="乱数とboosting_type変えて学習してソフトアンサンブルするか",
    )
    args = vars(ap.parse_args())

    with open(args["YML"]) as file:
        config = yaml.safe_load(file.read())

    # df_all = pd.read_csv(f"{DATA_DIR}/eng1.csv", index_col=0)
    df_all = pd.read_csv(args["INPUT_CSV"], index_col=0)
    if args["n_trials"] != 0:
        tuning(df_all, args["OUT_MODEL"], n_trials=args["n_trials"])
    else:
        if args["is_ensemble"]:
            train_soft_ensemble(df_all, args["OUT_MODEL"], config)
        else:
            train_select_cols(df_all, args["OUT_MODEL"], config)
