#!/bin/sh

start_dir=$(pwd)
cd $(dirname $1)

path=$(pwd) # get full path
basefile=$(basename $1 .csv)

cd $start_dir

echo "--------------- mlsvm-classifier -------------"

cd mlsvm/src

time /bin/sh -c "./mlsvm_csv_petsc --ds_p $path/ --ds_f $basefile  && \
     ./mlsvm_zscore --ds_p $path/ --ds_f $basefile && \
     ./mlsvm_knn --ds_p $path/ --ds_f $basefile"

cd $start_dir

echo
echo "--------------- mlsvm-KaHIP -------------"

cd multilevelSVM/

time optimized_output/prepare $path/${basefile}.csv

cd $start_dir
