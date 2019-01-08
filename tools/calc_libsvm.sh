#!/bin/sh

start_dir=$(pwd)
data_dir=$1

cd $data_dir
data_path=$(pwd) # get absolute path

cd $start_dir

libsvm_dir="$HOME/libsvm"
cd $libsvm_dir

echo "--------------- libSVM ------------"

RESULT=0
timeout_cnt=0
for e in $(seq 0 4) ;
do
    for f in $(seq 0 4) ;
    do
        train_file=train_exp_${e}_kfold_${f}.libsvm
        test_file=test_exp_${e}_kfold_${f}.libsvm


        ${libsvm_dir}/tools/grid.py -out null -gnuplot null



        # model_file=exp_${e}_kfold_${f}.model

        # for c_log in $(seq -5 2 15) ;
        # do
        #     c=$(echo "scale=10 2 ^ $c_log"| bc)
        #     for g_log in $(seq -5 2 15) ;
        #     do
        #         g=$(echo "scale=10 2 ^ $g_log"| bc)
        #         timeout -v 4h ./svm_train -c $c -g $g train_file model_file
        #         RESULT=$?
        #         if [ $RESULT -ne 0 ]; then
        #             ((++timeout_cnt))
        #             break
        #         fi
        #     done
        #     if [ $RESULT -ne 0 ]; then
        #         break
        #     fi
        # done
        # if ((  )); then
        #     break
        # fi
    done
    if [ $RESULT -ne 0 ]; then
        break
    fi
done

cd $start_dir

# rm -rf $my_tmp
