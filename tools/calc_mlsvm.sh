#!/bin/sh

start_dir=$(pwd)
basefile=$(basename $1 .csv)

cd $(dirname $1)
path=$(pwd) # get absolute path
cd $start_dir

mlsvm_dir="$HOME/mlsvm/src"
my_tmp="$(mktemp -d /tmp/${basefile}.XXXX)"
mkdir -p $my_tmp
cd $my_tmp

mkdir temp
mkdir svm_models
cp "${mlsvm_dir}"/mlsvm_classifier ./
cp "${mlsvm_dir}"/params.xml ./

echo "--------------- mlsvm ------------"

./mlsvm_classifier --ds_p "$path/" --ds_f $basefile  2>&1

cd $start_dir

# rm -rf $my_tmp
