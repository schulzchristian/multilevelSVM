#!/bin/sh

start_dir=$(pwd)
basefile=$(basename $1 .csv)

cd $(dirname $1)
path=$(pwd) # get absolute path
cd $start_dir

echo "--------------- mulitlevelSVM -------------"

cd multilevelSVM/

(time optimized_output/mlsvm $path/$basefile)  2>&1

    #grep -e "full graph" \
	 #-e "coarse nodes" \
         #-e "init train" \
         #-e "real.*s$"

cd $start_dir
