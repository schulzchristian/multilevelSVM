#!/usr/bin/env bash

csv_to_libsvm=$HOME/bachelor/multilevelSVM/tools/csv_to_libsvm.py
start_dir=$(pwd)
echo start_dir \"$start_dir\"

for d in $( find -L . -name temp -d );
do
    echo dir $d
    cd $d

    for e in $( seq 0 4 );
    do
        echo exp${e}
        for f in $( seq 0 4 );
        do
            echo kfold${f}

            train_file=train_exp_${e}_kfold_${f}.libsvm
            test_file=test_exp_${e}_kfold_${f}.libsvm

            $csv_to_libsvm "1" < kfold_p_train_data_exp_${e}_fold_${f}_exp_0.1_data > $train_file
            $csv_to_libsvm "-1" < kfold_n_train_data_exp_${e}_fold_${f}_exp_0.1_data >> $train_file
            $csv_to_libsvm < kfold_test_data_exp_${e}_fold_${f}_exp_0.1_data > $test_file
        done
    done
    echo back to $start_dir
    cd $start_dir
done
