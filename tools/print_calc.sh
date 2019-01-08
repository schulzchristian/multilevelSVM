dir=${1-data}
log_dir=${2-log}
import_dir=${3-log}

if [ "$3" ]; then
    echo "using imports at" "${import_dir}/file/temp"
fi

function calc_my {
for f in $(find data -type f -name "*.csv" -ls | sort -n -k7 |  awk '{ print $NF }') ; do
    FILE=$(basename $f .csv)
    if [ -z "$import_dir" ]; then
        CMD1="./multilevelSVM/tools/calc.sh $f > ./$log_dir/calc_${FILE} 2>&1 "
    else
        CMD1="./multilevelSVM/tools/calc.sh ${import_dir}/${FILE}/temp/ > ./$log_dir/calc_${FILE} 2>&1 "
    fi
    echo "echo \"$CMD1\" && $CMD1"
done
}

function calc_mlsvm {
for f in $(find data -type f -name "*.csv" -ls | sort -n -k7 |  awk '{ print $NF }') ; do
    FILE=$(basename $f .csv)
    CMD2="./multilevelSVM/tools/calc_mlsvm.sh $f > ./$log_dir/calc_${FILE}_mlsvm 2>&1 "
    #echo "screen -dm bash -c \"$CMD2\""
    echo "echo \"$CMD2\" && $CMD2"
done
}

function calc_all {
for f in $(find data -type f -name "*.csv" -ls | sort -n -k7 |  awk '{ print $NF }') ; do
    FILE=$(basename $f .csv)
    CMD1="./multilevelSVM/tools/calc.sh $f > ./$log_dir/calc_${FILE} 2>&1 "
    echo "echo \"$CMD1\" && $CMD1"
    CMD2="./multilevelSVM/tools/calc_mlsvm.sh $f > ./$log_dir/calc_${FILE}_mlsvm 2>&1 "
    echo "echo \"$CMD2\" && $CMD2"
done
}

mkdir -p $log_dir

calc_my > calc_my.sh

calc_mlsvm > calc_mlsvm.sh

calc_all > calc_all.sh
