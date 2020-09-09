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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

ORIG = r"C:\Users\81908\jupyter_notebook\tf_2_work\Probspace_geme_compe\data\orig"
train = pd.read_csv(f"{ORIG}/train_data.csv")
test = pd.read_csv(f"{ORIG}/test_data.csv")
df_all = train_df.append(test_df).reset_index(drop=True)

# + code_folding=[31, 42, 168]
import os
import random
import joblib

# tensorflowの警告抑制
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    ReLU,
    PReLU,
    Activation,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.model_selection import *
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ModelNN:
    def __init__(self, run_fold_name="", params={}) -> None:
        """コンストラクタ
        :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        :param params: ハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None
        self.scaler = None

    def build_model(self, input_shape):
        """モデル構築"""
        model = Sequential()
        model.add(Dense(self.params["units"][0], input_shape=input_shape))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(self.params["dropout"][0]))

        for l_i in range(1, self.params["layers"] - 1):
            model.add(Dense(self.params["units"][l_i]))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(self.params["dropout"][l_i]))

        model.add(Dense(self.params["nb_classes"]))
        model.add(Activation(self.params["pred_activation"]))
        if self.params["optimizer"] == "adam":
            opt = Adam(learning_rate=self.params["learning_rate"])
        else:
            opt = SGD(
                learning_rate=self.params["learning_rate"], momentum=0.9, nesterov=True
            )

        model.compile(
            loss=self.params["loss"], metrics=self.params["metrics"], optimizer=opt,
        )
        self.model = model

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # 乱数固定
        ModelNN().set_tf_random_seed()

        # 出力ディレクトリ作成
        os.makedirs(self.params["out_dir"], exist_ok=True)

        # データのセット・スケーリング
        validation = va_x is not None
        scaler = self.params["scaler"]  # StandardScaler()
        scaler.fit(tr_x)
        tr_x = scaler.transform(tr_x)
        # ラベルone-hot化
        tr_y = to_categorical(tr_y, num_classes=self.params["nb_classes"])

        # モデル構築
        self.build_model((tr_x.shape[1],))
        
        hist = None
        if validation:
            va_x = scaler.transform(va_x)
            va_y = to_categorical(va_y, num_classes=self.params["nb_classes"])

            cb = []
            cb.append(
                ModelCheckpoint(
                    filepath=os.path.join(
                        self.params["out_dir"], f"best_val_loss_{self.run_fold_name}.h5"
                    ),
                    monitor="val_loss",
                    save_best_only=True,
                    #verbose=1,
                    verbose=0,
                )
            )
            # cb.append(ModelCheckpoint(filepath=os.path.join(self.params["out_dir"], f"best_val_acc_{self.run_fold_name}.h5"),
            #        monitor="val_acc",
            #        save_best_only=True,
            #        verbose=1,
            #        mode="max",
            #    )
            # )
            cb.append(
                EarlyStopping(
                    monitor="val_loss", patience=self.params["patience"], verbose=1
                )
            )
            hist = self.model.fit(
                tr_x,
                tr_y,
                epochs=self.params["nb_epoch"],
                batch_size=self.params["batch_size"],
                #verbose=2,
                verbose=0,
                validation_data=(va_x, va_y),
                callbacks=cb,
            )
        else:
            hist = self.model.fit(
                tr_x,
                tr_y,
                epochs=self.params["nb_epoch"],
                batch_size=self.params["batch_size"],
                #verbose=2,
                verbose=0,
            )

        # スケーラー保存
        self.scaler = scaler
        joblib.dump(
            self.scaler,
            os.path.join(self.params["out_dir"], f"{self.run_fold_name}-scaler.pkl"),
        )
        
        # history plot
        self.plot_hist_acc_loss(hist)
        
        return hist

    def predict_binary(self, te_x):
        """2値分類の1クラスのみ取得"""
        self.load_model()
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict(te_x)[:, 1]
        return pred

    def load_model(self):
        model_path = os.path.join(
            self.params["out_dir"], f"best_val_loss_{self.run_fold_name}.h5"
        )
        # model_path = os.path.join(self.params['out_dir'], f'best_val_acc_{self.run_fold_name}.h5')
        scaler_path = os.path.join(
            self.params["out_dir"], f"{self.run_fold_name}-scaler.pkl"
        )
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"INFO: \nload model:{model_path} \nload scaler: {scaler_path}")

    def plot_hist_acc_loss(self, history):
        """学習historyをplot"""
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        # 1) Accracy Plt
        plt.plot(epochs, acc, 'bo' ,label = 'training acc')
        plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
        plt.title('Training and Validation acc')
        plt.legend()
        plt.savefig(f"{self.params['out_dir']}/{self.run_fold_name}-acc.png", bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()

        # 2) Loss Plt
        plt.plot(epochs, loss, 'bo' ,label = 'training loss')
        plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.savefig(f"{self.params['out_dir']}/{self.run_fold_name}-loss.png", bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()
        
    @staticmethod
    def set_tf_random_seed(seed=0):
        """
        tensorflow v2.0の乱数固定
        https://qiita.com/Rin-P/items/acacbb6bd93d88d1ca1b
        ※tensorflow-determinism が無いとgpuについては固定できないみたい
         tensorflow-determinism はpipでしか取れない($ pip install tensorflow-determinism)ので未確認
        """
        ## ソースコード上でGPUの計算順序の固定を記述
        # from tfdeterminism import patch
        # patch()
        # 乱数のseed値の固定
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)  # v1.0系だとtf.set_random_seed(seed)
    


# + code_folding=[5]
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import *

class Encoder():
    def __init__(self, 
                 encoder_flags={"count": True, "target": True, "catboost": True, "label": True, "impute_null": True}) -> None:
        self.encoder_flags = encoder_flags
    
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

    @staticmethod
    def impute_null_add_flag_col(df, strategy="median", cols_with_missing=None, fill_value=None):
        """欠損値を補間して欠損フラグ列を追加する
        fill_value はstrategy="constant"の時のみ有効になる補間する定数
        """
        from sklearn.impute import SimpleImputer

        df_plus = df.copy()

        if cols_with_missing is None:
            if strategy in ["median", "median"]:
                # 数値列で欠損ある列探す
                cols_with_missing = [col for col in df.columns if (df[col].isnull().any()) and (df[col].dtype.name not in ["object", "category", "bool"])]
            else:
                # 欠損ある列探す
                cols_with_missing = [col for col in df.columns if (df[col].isnull().any())]

        for col in cols_with_missing:
            # 欠損フラグ列を追加
            #df_plus[col + "_was_missing"] = df[col].isnull()
            #df_plus[col + "_was_missing"] = df_plus[col + "_was_missing"].astype(int)
            # 欠損値を平均値で補間
            my_imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            df_plus[col] = my_imputer.fit_transform(df[cols_with_missing])

        return df_plus
    
    def run_encoders(self, X, y, train_index, val_index, test_df=None):
        """カウント,ターゲット,CatBoost,ラベルエンコディング一気にやる。cvの処理のforの中に書きやすいように"""
        train_df = pd.concat([X, y], axis=1)
        t_fold_df, v_fold_df = train_df.iloc[train_index], train_df.iloc[val_index]

        if self.encoder_flags["count"]:
            # カウントエンコディング
            t_fold_df, v_fold_df = Encoder().count_encoder(t_fold_df, v_fold_df, cat_features=None)
            if df_test is None:
                _, test_df = Encoder().count_encoder(t_fold_df, test_df, cat_features=None)
        if self.encoder_flags["target"]:
            # ターゲットエンコディング
            t_fold_df, v_fold_df = Encoder().target_encoder(t_fold_df, v_fold_df, target_col=y.name, cat_features=None)
        if self.encoder_flags["catboost"]:
            # CatBoostエンコディング
            t_fold_df, v_fold_df = Encoder().catboost_encoder(t_fold_df, v_fold_df, target_col=y.name, cat_features=None)
        
        if self.encoder_flags["label"]:
            # ラベルエンコディング
            train_df = t_fold_df.append(v_fold_df)  # trainとval再連結
            cate_cols = t_fold_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
            for col in cate_cols:
                train_df[col], uni = pd.factorize(train_df[col])
                
        if self.encoder_flags["impute_null"]:
            # 欠損置換（ラベルエンコディングの後じゃないと処理遅くなる）
            nulls = train_df.drop(y.name, axis=1).isnull().sum().to_frame()
            null_indexs = [index for index, row in nulls.iterrows() if row[0] > 0]
            train_df = Encoder().impute_null_add_flag_col(train_df, cols_with_missing=null_indexs, strategy="most_frequent")  # 最頻値で補間
        
        t_fold_df, v_fold_df = train_df.iloc[train_index], train_df.iloc[val_index]
        print(
            "run encoding Train shape: {}, valid shape: {}".format(
                t_fold_df.shape, v_fold_df.shape
            )
        )
        feats = t_fold_df.columns.to_list()
        feats.remove(y.name)
        X_train, y_train = (t_fold_df[feats], t_fold_df[y.name])
        X_val, y_val = (v_fold_df[feats], v_fold_df[y.name])
        return X_train, y_train, X_val, y_val



# + code_folding=[65]
import warnings
warnings.filterwarnings('ignore')

def prepare_data(train, test, target_col):
    """前処理"""
    df = train.copy()
    df = df.append(test)
    df = df.drop(["id", "game-ver"], axis=1)
    
    ## 時刻ばらす periodそのまま残すほうがcv acc上がる
    #df["period"] = pd.to_datetime(df["period"])
    #df['year'] = df["period"].dt.year
    #df['month'] = df["period"].dt.month
    #df['dayofyear'] = df["period"].dt.dayofyear
    #df['dayofweek'] = df["period"].dt.dayofweek
    #df['weekend'] = (df["period"].dt.dayofweek.values >= 5).astype(int)
    #df['hour'] = df["period"].dt.hour
    #df = df.drop(["period"], axis=1)
    
    # カテゴリ列保持
    cat_features = df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    print("cat_features", cat_features)
    
    # ラベルエンコディング
    cate_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    for col in cate_cols:
        df[col], uni = pd.factorize(df[col])
    
    # 欠損置換（ラベルエンコディングの後じゃないと処理遅くなる）
    nulls = df.drop(target_col, axis=1).isnull().sum().to_frame()
    null_indexs = [index for index, row in nulls.iterrows() if row[0] > 0]
    df = Encoder().impute_null_add_flag_col(df, cols_with_missing=null_indexs, strategy="most_frequent")  # 最頻値で補間
    
    display(df.head(2))
    
    train = df[df[target_col].notnull()]  # 欠損ではない行のみ
    test = df[df[target_col].isnull()]  # 欠損行のみ
    test = test.drop(target_col, axis=1)
    
    return train, test, cat_features


def enc_df(df_tr, df_va, df_te, target_col, cat_features):
    """
    train/valid/testのtarget_encoderとか一括実行
    エンコードで使うtrainのデータはvalid,test同じにしないとおかしくなるので
    """
    print("before enc_df:", df_tr.shape, df_va.shape, df_te.shape)
    # エンコードの基準のデータフレーム
    df_base = df_tr.copy()
    
    # train用
    df_tr, df_va = Encoder().count_encoder(df_base, df_va, cat_features=cat_features)
    #df_tr, df_va = Encoder().target_encoder(df_tr, df_va, target_col=target_col, cat_features=cat_features)
    df_tr, df_va = Encoder().catboost_encoder(df_tr, df_va, target_col=target_col, cat_features=cat_features)
    
    # submit用
    _df_tr, df_te = Encoder().count_encoder(df_base, df_te, cat_features=cat_features)
    #_df_tr, df_te = Encoder().target_encoder(_df_tr, df_te, target_col=target_col, cat_features=cat_features)
    _df_tr, df_te = Encoder().catboost_encoder(_df_tr, df_te, target_col=target_col, cat_features=cat_features)
    
    print("after enc_df:", df_tr.shape, df_va.shape, df_te.shape)
    return df_tr, df_va, df_te
    

def submit_cv_csv(test_preds, out_dir):
    """cvで出したtestの確信度のリストを平均してsubmit.csv出力"""
    te_mean = pd.DataFrame(test_preds).apply(lambda x: np.mean(x), axis=1).values
    te_mean[te_mean >= 0.5] = 1
    te_mean[te_mean < 0.5] = 0
    te_mean = te_mean.astype(int)
    output_csv = f"{out_dir}/submission_kernel.csv"
    pd.DataFrame({"id": range(len(te_mean)), "y": te_mean}).to_csv(
        output_csv, index=False
    )
    print(f"INFO: save csv {output_csv}")
    
    
def train_nn_cv(df_train, df_test, feats, target_col, params, encoder_flags, num_folds=2, cat_features=None):
    """cvでnnモデル学習してsubmit.csv出力"""
    #folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    
    test_preds = {}
    fold_mean_acc = 0.0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train[target_col])):
        print("n_fold:", n_fold)
    
        # target_encoderとか
        df_tr, df_va, df_te = enc_df(df_train.loc[train_idx], df_train.loc[valid_idx], df_test, target_col, cat_features)
        train_x, train_y = df_tr[feats], df_tr[target_col]
        valid_x, valid_y = df_va[feats], df_va[target_col]
        test_x = df_te[feats]
        
        # model
        model_cls = ModelNN(n_fold, params)
        
        # 学習
        hist = model_cls.train(train_x, train_y, valid_x, valid_y)
        
        # 予測
        valid_pred = model_cls.predict_binary(valid_x)
        print("valid_pred:", valid_pred)

        # 正解率
        valid_pred[valid_pred >= 0.5] = 1
        valid_pred[valid_pred < 0.5] = 0
        fold_acc = accuracy_score(valid_y, valid_pred)
        fold_mean_acc += fold_acc / num_folds
        print(f"fold={num_folds}, acc={fold_acc}\n")
        
        # submitの確信度
        test_pred = model_cls.predict_binary(test_x)
        test_preds[n_fold] = test_pred
        print("test_pred:", test_pred, "\n")
        
    # submit.csv出力
    submit_cv_csv(test_preds, params["out_dir"])
    
    return fold_mean_acc
    
        
if __name__ == '__main__':
    
    target_col = "y"
    df_train, df_test, cat_features = prepare_data(train, test, target_col)
    #display(df_test)
    
    feats = df_test.columns.to_list()
    params = dict(
        out_dir="tmp",
        #scaler=MinMaxScaler(),
        scaler=StandardScaler(),
        layers=3,
        units=[128, 64, 32],
        dropout=[0.3, 0.3, 0.3],
        nb_classes=2,
        pred_activation="softmax",
        loss="categorical_crossentropy",
        optimizer="adam",
        learning_rate=0.001,
        metrics=["acc"],
        #nb_epoch=10,
        #patience=3,
        nb_epoch=100,
        patience=30,
        batch_size=256,
    )
    num_folds = 5
    
    fold_mean_acc = train_nn_cv(df_train, df_test, feats, target_col, params, encoder_flags, num_folds=num_folds, cat_features=cat_features)
    print("fold_mean_acc", fold_mean_acc)
# -
















