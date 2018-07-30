#!/bin/sh

start_dir=$(pwd)
basefile=$(basename $1 .csv)

echo "--------------- mulitlevelSVM -------------"

#~/multilevelSVM/optimized_output/mlsvm $path/$basefile -e 5 -f 5 --import_kfold 2>&1
~/multilevelSVM/optimized_output/mlsvm -e 5 -f 5 --import_kfold $1 2>&1

cd $start_dir
