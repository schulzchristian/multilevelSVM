dir=${1-data}
log_dir=${2-log}
import_dir=${3}


if [ "$import_dir" ]; then
    echo "using imports at" "${import_dir}/<file>/temp"
    import="true"
fi
if [ -z "$import_dir" ]; then
   echo "no import"
fi

files=$(find data -name "*.csv" -ls | sort -n -k7 |  awk '{ print $NF }')

params=" -e 5 -k 5 -c 32 "
params="$params --matching=low_diameter"

function calc_my {
for f in $files ; do
    file=$(basename $f .csv)
    output="./$log_dir/calc_${file}"
    _params=$params
    if [ -z "$import_dir" ]; then
        input="$dir/$file"
    else
        input="${import_dir}/${file}/temp/"
	_params="$_params --validation=kfold_import"
    fi
    echo "./multilevelSVM/optimized_output/mlsvm ${_params} ${input} > ${output} 2>&1 "
done
}

function calc_single {
for f in $files ; do
    file=$(basename $f .csv)
    output="./$log_dir/calc_${file}_single 2>&1"
    _params=$params
    if [ -z "$import_dir" ]; then
        input="$dir/$file"
    else
        input="${import_dir}/${file}/temp/"
	_params="$_params --validation=kfold_import"
    fi
    echo "./multilevelSVM/optimized_output/single_level_svm ${_params} ${input} > ${output} 2>&1 "
done
}

function calc_mlsvm {
for f in $files ; do
    file=$(basename $f .csv)
    output="./$log_dir/calc_${file} 2>&1"
    echo "./multilevelSVM/tools/calc_mlsvm.sh $f > ./$log_dir/calc_${file}_mlsvm 2>&1 "
done
}

mkdir -p $log_dir

echo "set -x" > calc_my.sh
calc_my >> calc_my.sh

echo "set -x" > calc_single.sh
calc_single >> calc_single.sh

#calc_mlsvm > calc_mlsvm.sh

paste -d '\n' calc_my.sh calc_single.sh > calc_all.sh
