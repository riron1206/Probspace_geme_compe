"""
特徴量選択
Usage:
    $ conda activate tfgpu
    $ python ./feature_select.py -o ../data/feature_select/eng2_csv -i ../data/feature_eng/eng2.csv
    $ python ./feature_select.py -o ../data/feature_select/eng3_csv -i ../data/feature_eng/eng3.csv
    $ python ./feature_select.py -o ../data/feature_select/code_eng1_1_csv -i ../data/code_feature_eng/eng1_1.csv # 2020/08/30
    $ python ./feature_select.py -o ../data/feature_select/code_eng3_csv -i ../data/code_feature_eng/eng3.csv
    $ python ./feature_select.py -o ../data/feature_select/code_eng3_csv -i ../data/code_feature_eng/eng3.csv -m feature_selection
    $ python ./feature_select.py -o ../data/feature_select/code_eng3_csv -i ../data/code_feature_eng/eng3.csv -m remove_useless_features
"""
import os
import sys
import warnings
import argparse
from functools import partial

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import *
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(r"C:\Users\81908\Git\xfeat")
import xfeat
from xfeat import *
from xfeat.selector import *
from xfeat.utils import *

sys.path.append(r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\code")
from train import Model

sns.set()
warnings.filterwarnings("ignore")
matplotlib.use("Agg")


class FeatureSelect:
    @staticmethod
    def xfeat_feature_explored_clf(
        params,
        out_dir,
        dict_enc_flag={"count": True, "target": True, "catboost": True},
        n_trials=50,
        threshold_range=(0.3, 1.0),  # 特徴量減らす割合の最小値最大値
        selecter_objective="binary",  # 特徴量選択するlgbのobjective
        selecter_metric="binary_logloss",  # 特徴量選択するlgbのmetric
    ):
        """
        xfeatのGBDTFeatureExplorerで特徴量削減する
        kfold_cv_LGBMClassifier()を使うので2値分類のみ可能
        """

        def objective(selector, params, trial):
            # selected columns
            selector.set_trial(trial)
            selector.fit(tr_df)
            input_cols = selector.get_selected_cols()
            print(f"input_n_cols: {len(input_cols)}")
            # print(f"input_cols: \n{input_cols}")

            # feture importance高かった列だけ採用
            params["select_cols"] = input_cols

            # Evaluate with selected columns
            mean_fold_score, _, _ = model.kfold_cv_LGBMClassifier(**params)
            return mean_fold_score

        os.makedirs(out_dir, exist_ok=True)
        model = Model(out_dir, dict_enc_flag)
        target_col = params["target_col"]

        ################ 特徴量準備 ################
        # ターゲットエンコディングとかする
        _df = params["df"].copy()
        _df = _df[_df[target_col].notnull()].reset_index(drop=True)  # test set取り除く
        if params["stratified"]:
            folds = StratifiedKFold(
                n_splits=params["num_folds"], shuffle=True, random_state=1001
            )
        else:
            folds = KFold(n_splits=params["num_folds"], shuffle=True, random_state=1001)
        tr_df, te_df = None, None
        for n_fold, (train_idx, valid_idx) in enumerate(
            folds.split(_df.drop(target_col, axis=1), _df[target_col])
        ):
            # ターゲットエンコディングとか一括実行
            tr_df, te_df = model._encoding(
                _df.iloc[train_idx], _df.iloc[valid_idx], target_col
            )
            break
        # display(tr_df.head())

        ################ GBDTFeatureExplorer ################
        input_cols = tr_df.columns.to_list()
        input_cols.remove(target_col)

        _lgbm_params = params["lgb_params"].copy()
        _lgbm_params["objective"] = selecter_objective
        _lgbm_params["metric"] = selecter_metric
        _lgbm_params["verbosity"] = -1

        # xfeatの特徴量選択用lgmモデルを返す(threshold_rangeのbestな値を探索するための)
        selector = GBDTFeatureExplorer(
            input_cols=input_cols,
            target_col=target_col,
            fit_once=True,
            threshold_range=threshold_range,
            lgbm_params=_lgbm_params,
            lgbm_fit_kwargs={"num_boost_round": 1000},
        )

        ################ 探索実行　################
        study = optuna.create_study(
            study_name="feature_explored",
            storage=f"sqlite:///{out_dir}/feature_explored.db",
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=1),
        )
        study.optimize(
            partial(objective, selector, params),
            n_trials=n_trials,
            n_jobs=1,
            gc_after_trial=True,
        )
        print(f"\nstudy.best_params:\n{study.best_params}")
        print(f"\nstudy.best_trial:\n{study.best_trial}")
        pd.DataFrame(
            {f"best_score({params['eval_metric']})": [study.best_value]}
        ).to_csv(f"{out_dir}/xfeat_feature_explored_clf_best_score.tsv", sep="\t")

        # 探索履歴保存
        study.trials_dataframe().to_csv(
            f"{out_dir}/feature_explored_history.csv", index=False
        )

        # bestな選択をした特徴量
        selector.from_trial(study.best_trial)
        selected_cols = selector.get_selected_cols()
        print(f" - {len(input_cols) - len(selected_cols)} features are removed.")
        pd.DataFrame({"selected_cols": selected_cols}).to_csv(
            f"{out_dir}/xfeat_feature_explored_clf_selected_cols.csv"
        )

    @staticmethod
    def xfeat_remove_useless_features(df, cols=None, threshold=0.8):
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
        return df_reduced

    @staticmethod
    def xfeat_feature_selection(
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

    @staticmethod
    def rfecv_kfold(X_train, y_train, sk_model, out_dir, scoring="accuracy"):
        """RFECV(交差検証+再帰的特徴除去)"""

        def _plot_rfecv(selector):
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (nb of correct classifications)")
            plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
            plt.savefig(f"{out_dir}/plot_rfecv.png")

        # RFECVは交差検証+再帰的特徴除去。データでかいとメモリ死ぬので注意
        # RFE（再帰的特徴除去=recursive feature elimination: すべての特徴量を使う状態から、1つずつ特徴量を取り除いていく）で特徴量選択
        selector = RFECV(
            sk_model, cv=KFold(3, shuffle=True), scoring=scoring, n_jobs=-1
        )
        selector.fit(X_train, y_train)
        # 探索履歴plot
        _plot_rfecv(selector)
        # 選択した特徴量
        select_cols = X_train.columns[selector.get_support()].to_list()
        print("\nselect_cols:\n", select_cols, len(select_cols))
        # 捨てた特徴量
        print("not select_cols:\n", X_train.columns[~selector.get_support()].to_list())
        # 選択した特徴量保存
        select_cols.append("y")
        pd.DataFrame({"select_cols": select_cols}).to_csv(
            f"{out_dir}/rfecv_select_cols.csv", index=False
        )


def run_xfeat_feature_explored_clf(df, out_dir, target_col="y", n_trials=200):
    """xfeat_feature_explored_clf()実行"""
    best_params = {
        "subsample": 0.9,
        "subsample_freq": 6,
        "colsample_bytree": 0.1,
        "max_depth": 7,
        "min_child_samples": 343,
        "min_child_weight": 0.04084861948055769,
        "num_leaves": 95,
        "reg_alpha": 0.5612212694825488,
        "reg_lambda": 0.0001757886119766502,
    }
    lgb_params = dict(
        n_estimators=10000,
        learning_rate=0.01,
        silent=-1,
        importance_type="gain",
        random_state=71,
        **best_params,
    )
    params = dict(
        lgb_params=lgb_params,
        df=df,
        num_folds=4,
        target_col=target_col,
        del_cols=None,
        eval_metric="error",
        # stratified=True,
        stratified=False,
        is_submission=False,
        is_plot_perm_importance=False,
    )
    FeatureSelect().xfeat_feature_explored_clf(
        params,
        out_dir,
        dict_enc_flag={"count": False, "target": True, "catboost": True},
        n_trials=n_trials,
        threshold_range=(0.1, 1.0),
    )


def run_feature_selection(df, out_dir, target_col="y", n_cols=300):
    """xfeat_feature_selection()実行"""
    # target encording
    df = df[df[target_col].notnull()]
    (X_train, X_test, y_train, y_test) = train_test_split(
        df.drop(target_col, axis=1), df[target_col], test_size=0.1, random_state=71
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_test, y_test], axis=1)
    train_df, _ = Model()._encoding(train_df, valid_df, target_col)
    print(train_df.shape)

    # feature_importance高い順に列数を 列数*threshold にする
    threshold = n_cols / train_df.shape[1]
    params = {
        "metric": "binary_logloss",
        "objective": "binary",
        "threshold": threshold,
    }  # metric=roc_aucでも可能
    select_df = FeatureSelect().xfeat_feature_selection(train_df, target_col, params)
    print(select_df.shape)
    print(select_df.columns)

    # 列名保持
    feature_selections = sorted(select_df.columns.to_list())
    feature_selections.append(target_col)
    pd.DataFrame({"feature_selections": feature_selections}).to_csv(
        f"{out_dir}/feature_selection_cols.csv", index=False
    )

    not_feature_selections = list(
        set(train_df.columns.to_list()) - set(feature_selections)
    )
    pd.DataFrame({"not_feature_selections": not_feature_selections}).to_csv(
        f"{out_dir}/not_feature_selection_cols.csv", index=False
    )


def run_rfecv_clf(df, out_dir, target_col="y"):
    """rfecv_kfold()実行"""
    X_train = df.loc[df[target_col].notnull()]
    # label encoding
    cate_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.to_list()
    for col in cate_cols:
        X_train[col], uni = pd.factorize(X_train[col])
    y_train = X_train[target_col]
    best_params = {
        "bagging_fraction": 0.9,
        "bagging_freq": 6,
        "feature_fraction": 0.1,
        "max_depth": 7,
        "min_child_samples": 343,
        "min_child_weight": 0.04084861948055769,
        "num_leaves": 95,
        "reg_alpha": 0.5612212694825488,
        "reg_lambda": 0.0001757886119766502,
    }
    clf = lgb.LGBMClassifier(n_jobs=-1, seed=71, **best_params)
    FeatureSelect().rfecv_kfold(X_train, y_train, clf, out_dir, scoring="accuracy")


def test_func():
    """
    テスト駆動開発での関数のテスト関数
    test用関数はpythonパッケージの nose で実行するのがおすすめ($ conda install -c conda-forge nose などでインストール必要)
    →noseは再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行する
    $ cd <このモジュールの場所>
    $ nosetests -v -s <本モジュール>.py  # 再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行。-s付けるとprint()の内容出してくれる
    """
    import seaborn as sns

    def _test_run_xfeat_feature_explored_clf():
        """testデータ(タイタニック)でrun_xfeat_feature_explored_clf()実行"""
        df = sns.load_dataset("titanic")
        df = df.drop(["alive"], axis=1)
        run_xfeat_feature_explored_clf(df, "tmp", target_col="survived", n_trials=2)

    def _test_xfeat_remove_useless_features():
        """testデータ(タイタニック)でxfeat_remove_useless_features()実行"""
        df = sns.load_dataset("titanic")
        df = df.drop(["alive"], axis=1)
        y_seri = df["survived"]
        df = df.drop("survived", axis=1)
        df = FeatureSelect().xfeat_remove_useless_features(df)
        df = pd.concat([df, y_seri], axis=1)
        df.to_csv(f"tmp/remove_useless_features.csv")

    def _test_run_feature_selection():
        """testデータ(タイタニック)でrun_feature_selection()実行"""
        df = sns.load_dataset("titanic")
        df = df.drop(["alive"], axis=1)
        run_feature_selection(df, "tmp", target_col="survived", n_cols=10)

    def _test_run_rfecv_clf():
        """testデータ(タイタニック)でrun_rfecv_clf()実行"""
        df = sns.load_dataset("titanic")
        df = df.drop(["alive"], axis=1)
        run_rfecv_clf(df, "tmp", target_col="survived")

    # _test_run_xfeat_feature_explored_clf()
    # _test_xfeat_remove_useless_features()
    # _test_run_feature_selection()
    _test_run_rfecv_clf()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o", "--OUT_DIR", type=str,
    )
    ap.add_argument(
        "-i", "--INPUT_CSV", type=str,
    )
    ap.add_argument(
        "-m", "--mode", type=str, default="feature_explored", help="どの特徴量選択を実行するか"
    )
    args = vars(ap.parse_args())

    df = pd.read_csv(
        # r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\orig\train_data.csv"
        # r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\feature_eng\eng2.csv"
        args["INPUT_CSV"],
        index_col=0,
    )
    if args["mode"] == "feature_explored":
        run_xfeat_feature_explored_clf(df, args["OUT_DIR"], target_col="y")

    elif args["mode"] == "remove_useless_features":
        y_seri = df["y"]
        df = df.drop("y", axis=1)
        df_select = FeatureSelect().xfeat_remove_useless_features(df)
        df = pd.concat([df[df_select.columns], y_seri], axis=1)
        df.to_csv(f"{args['OUT_DIR']}/remove_useless_features.csv")

    elif args["mode"] == "feature_selection":
        run_feature_selection(df, args["OUT_DIR"], target_col="y")

    elif args["mode"] == "rfecv":
        run_rfecv_clf(df, args["OUT_DIR"], target_col="y")
