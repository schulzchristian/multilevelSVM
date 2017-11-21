#!/bin/sh

start_dir=$(pwd)
cd $(dirname $1)

path=$(pwd) # get full path
file=$(basename $1 .csv)

start_dir=$(pwd)

echo "--------------- mlsvm------------"

cd mlsvm/src

(time ./mlsvm_classifier --ds_p $path/ --ds_f ${file}) 2>&1 | \
    grep -e "num points" \
         -e "initial training" \
         -e "\[SV\]\[ETD\]" \
         -e "real.*s$"

cd $start_dir

echo
echo "--------------- mlsvm-KaHIP -------------"

cd multilevelSVM/
(time optimized_output/mlsvm ${1}) 2>&1 | \
    grep -e "coarse nodes" \
         -e "init train" \
         -e "real.*s$"

cd $start_dir
