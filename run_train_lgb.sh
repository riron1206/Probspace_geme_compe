#!/bin/bash
PWDDIR=`pwd`

PYTHON=/c/Users/81908/anaconda3/envs/tfgpu/python.exe
PY=$PWDDIR/code/train_lgb.py

conda activate tfgpu

cd code

eng2_csv_best_params1() {
    # LB=0.552294(submission_kernel.csv) 2020/08/29
    $PYTHON $PY \
        -o ../model/code_train/eng2_csv \
        -i ../data/feature_eng/20200828/eng2.csv \
        -y config/best_params1.yml
}

eng1_csv_best_params_eng2_csv_tuining() {
    # LB=0.550318(submission_kernel.csv) cv_error=0.445442 2020/08/29
    $PYTHON $PY \
        -o ../model/code_train/eng1_csv \
        -i ../data/feature_eng/eng1.csv \
        -y config/best_params_eng2_csv_tuining.yml
}

eng2_csv_best_params_eng2_csv_tuining() {
    # LB=0.550600(submission_kernel.csv) cv_error=0.444777 2020/08/29
    $PYTHON $PY \
        -o ../model/code_train/eng2_csv \
        -i ../data/feature_eng/eng2.csv \
        -y config/best_params_eng2_csv_tuining.yml
}

eng1_1_csv_best_params_eng2_csv_tuining() {
    # cv_error=0.442659 2020/08/29
    $PYTHON $PY \
        -o ../model/code_train/eng1_1_csv \
        -i ../data/feature_eng/eng1_1.csv \
        -y config/best_params_eng2_csv_tuining.yml
}

code_eng1_1_csv_best_params_eng2_csv_tuining() {
    # LB=0.550176(submission_kernel.csv) cv_error=0.437471  2020/08/29
    $PYTHON $PY \
        -o ../model/code_train/code_eng1_1_csv \
        -i ../data/code_feature_eng/eng1_1.csv \
        -y config/best_params_eng2_csv_tuining.yml
}

code_eng1_1_csv_best_params_eng1_1_csv_tuining() {
    # LB=0.554128(submission_kernel.csv) cv_error=0.440300  2020/09/09
    $PYTHON $PY \
        -o ../model/code_train/code_eng1_1_csv/train_best_param \
        -i ../data/code_feature_eng/eng1_1.csv \
        -y config/best_params_eng1_1_csv_tuining.yml
}

code_eng1_1_csv_best_params_eng1_1_csv_tuining_ensemble() {
    # LB=0.552999(gbdtのみのsubmission_ensemble.csv) cv_error=  2020/08/31
    $PYTHON $PY \
        -o ../model/code_train/code_eng1_1_csv/train_best_param_ensemble \
        -i ../data/code_feature_eng/eng1_1.csv \
        -y config/best_params_eng1_1_csv_tuining.yml \
        -is_e
}

code_eng2_csv_best_params_eng2_csv_tuining() {
    # LB=0.550459(submission_kernel.csv) cv_error=0.437773  2020/09/09
    $PYTHON $PY \
        -o ../model/code_train/code_eng2_csv/train_best_param \
        -i ../data/code_feature_eng/eng2.csv \
        -y config/best_params_eng2_csv_tuining.yml
}

code_eng3_remove_useless_features_csv_best_params_eng2_csv_tuining() {
    # cv_error=0.470615 2020/08/29
    $PYTHON $PY \
        -o ../model/code_train/code_eng3_remove_useless_features \
        -i ../data/feature_select/code_eng3_csv/remove_useless_features.csv \
        -y config/best_params_eng2_csv_tuining.yml
}

code_eng3_features_select_csv_best_params_eng1_1_csv_tuining_select_cols() {
    # cv_error= 2020/08/30
    $PYTHON $PY \
        -o ../model/code_train/code_eng3_features_select
        -i ../data/code_feature_eng/eng3.csv
        -y config/best_params_eng1_1_csv_tuining_select_cols.yml
}

# 1th 2020/09/09
#code_eng1_1_csv_best_params_eng1_1_csv_tuining;
# 2th 2020/08/31
#code_eng1_1_csv_best_params_eng1_1_csv_tuining_ensemble;

code_eng2_csv_best_params_eng2_csv_tuining;

# 元のディレクトリに戻る
cd ..