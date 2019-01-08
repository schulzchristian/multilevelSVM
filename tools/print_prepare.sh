for f in $(find ${1:-data} -name "*.csv") ; do
	CMD="./multilevelSVM/tools/prepare.sh $f > ./log/prepare_$(basename $f .csv) 2>&1"
	echo "echo \"$CMD\" && $CMD"
done
