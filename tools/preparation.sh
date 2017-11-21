#!/bin/sh

start_dir=$(pwd)
cd $(dirname $1)

path=$(pwd) # get full path
basefile=$(basename $1 .csv)

cd $start_dir

echo "--------------- mlsvm-classifier -------------"

cd mlsvm/src

time ./mlsvm_csv_petsc --ds_p $path/ --ds_f ${file}  &&
    ./mlsvm_zscore --ds_p $path/ --ds_f ${file} &&
    ./mlsvm_knn --ds_p $path/ --ds_f ${file}

cd $start_dir

echo
echo "--------------- mlsvm-KaHIP -------------"

cd multilevelSVM/

time optimized_output/csv_flann ${1}

cd $start_dir
