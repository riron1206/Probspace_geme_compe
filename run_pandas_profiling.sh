#!/bin/bash
PWDDIR=`pwd`

PYTHON=/c/Users/81908/anaconda3/envs/tfgpu/python.exe
PY=./code/summary_html_pandas_profiling.py
OUT=./data/eda
mkdir -p $OUT

#source conda activate tfgpu

$PYTHON $PY --output_dir $OUT --input_file ./data/orig/train_data.csv
$PYTHON $PY --output_dir $OUT --input_file ./data/orig/test_data.csv
$PYTHON $PY --output_dir $OUT --input_file ./data/orig/statink-weapon2.csv