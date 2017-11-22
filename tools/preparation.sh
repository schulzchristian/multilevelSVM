#!/bin/sh

start_dir=$(pwd)
cd $(dirname $1)

path=$(pwd) # get full path
file=$(basename $1)
basefile=${file%.*}

cd $start_dir

echo "--------------- mlsvm-classifier -------------"

cd mlsvm/src

time ./mlsvm_csv_petsc --ds_p $path/ --ds_f $basefile  && \
     ./mlsvm_zscore --ds_p $path/ --ds_f $basefile && \
     ./mlsvm_knn --ds_p $path/ --ds_f $basefile

cd $start_dir

echo
echo "--------------- mlsvm-KaHIP -------------"

cd multilevelSVM/
pwd

time optimized_output/csv_flann $path/${basefile}.csv

cd $start_dir
