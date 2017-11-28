#!/bin/sh

start_dir=$(pwd)
basefile=$(basename $1 .csv)

cd $(dirname $1)
path=$(pwd) # get absolute path
cd $start_dir

echo "--------------- mlsvm ------------"

cd mlsvm/src

(time ./mlsvm_classifier --ds_p $path/ --ds_f ${basefile})  2>&1

cd $start_dir
