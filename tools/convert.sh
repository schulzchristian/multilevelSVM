#!/bin/sh
set -e

mlsvm_dir="$HOME/mlsvm/src"
tools_dir="$HOME/multilevelSVM/tools"

cd $HOME/log_export

for d in $(find . -maxdepth 1 -type d -not -path ".") ; do
	echo "--------- $d ------------"
	cd $d

	rm *.info || true

	for f in $(find . -type f -name "*0.1") ; do
		echo $f
		${mlsvm_dir}/petsc_utility/get_mat_info -i $f | python ${tools_dir}/petsc_to_csv.py > ${f}_data
	done
done

cd $start_dir

# rm -rf $my_tmp
