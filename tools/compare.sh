#!/bin/sh

echo "--------------- mlsvm ------------"
grep -e "num points" \
     -e "initial training" \
     -e "\[SV\]\[ETD\]" \
     -e "real.*s$" \
     -e "user" \
     $1"_mlsvm"


echo "---------- multilevelSVM ---------"
grep -e "full graph" \
     -e "coarse nodes" \
     -e "init train" \
     -e "real.*s$" \
     -e "user" \
     $1
