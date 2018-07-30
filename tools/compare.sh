#!/bin/sh

echo "--------------- mlsvm ------------"
grep -e "num points" \
     -e "prepare_data_for_iteration" \
     -e "initial training" \
     -e "final TD" \
     -e "Vcycle" \
     -e "Final Results" \
     -e "\[CP\]\[PFR\]" \
     -e "\*   Acc" \
     -e "real.*s$" \
     -e "user" \
     $1"_mlsvm"


echo "---------- multilevelSVM ---------"
grep -e "coarse nodes" \
     -e "KFOLD_TIME" \
     -e "COARSE_TIME" \
     -e "INIT_TRAIN_TIME" \
     -e "INIT_ACC" \
     -e "INIT_GMEAN" \
     -e "TIME" \
     $1
