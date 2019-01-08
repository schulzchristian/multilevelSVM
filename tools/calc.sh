#!/bin/sh

start_dir=$(pwd)
basefile=$(basename $1 .csv)
path=$(dirname $1)

echo "--------------- mulitlevelSVM -------------"

#/multilevelSVM/optimized_output/mlsvm -e 5 -k 5 -b $path/$basefile 2>&1
~/multilevelSVM/optimized_output/mlsvm -e 5 -k 5 -b --matching=low_diameter --import_kfold $1 2>&1

cd $start_dir
