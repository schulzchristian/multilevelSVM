#include <argtable2.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <regex.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <memory>
#include <svm.h>

#include <flann/flann.hpp>
#include "data_structure/graph_access.h"
#include "io/graph_io.h"
#include "partition/partition_config.h"
#include "svm/svm_convert.h"
#include "svm/k_fold.h"
#include "svm/k_fold_build.h"
#include "svm/k_fold_import.h"
#include "svm/k_fold_once.h"
#include "svm/results.h"
#include "tools/timer.h"
#include "parse_parameters.h"

#include <thundersvm/util/log.h>
void print_null(const char *s) {}

#define SVM_SOLVER svm_solver_thunder
#define SVM_MODEL SVC

int main(int argn, char *argv[]) {
	PartitionConfig partition_config;
	
        if(parse_parameters(argn, argv, partition_config)) {
                return -1;
        }

	partition_config.apply();
	partition_config.print();

        results results;
	
        for (int exp = 0; exp < partition_config.num_experiments; exp++) {
        std::cout << " \\/\\/\\/\\/\\/\\/\\/\\/\\/ EXPERIMENT " << exp << " \\/\\/\\/\\/\\/\\/\\/" << std::endl;

        std::unique_ptr<k_fold> kfold;

	switch (partition_config.validation_type) {
	case KFOLD:
		kfold.reset(new k_fold_build(partition_config,
					     partition_config.filename));
		break;
	case KFOLD_IMPORT:
		kfold.reset(new k_fold_import(partition_config, exp,
					      partition_config.filename));
		break;
	case ONCE:
		kfold.reset(new k_fold_once(partition_config,
					    partition_config.filename));
		break;
	// case TRAIN_TEST_SPLIT:
		// kfold.reset(new k_fold_traintest(partition_config,
		// 				 partition_config.filename,
		// 				 partition_config.testname));
	}

        timer t_all;
        timer t;
        double kfold_io_time = 0;

        while (kfold->next(kfold_io_time)) {
        results.next();

        auto kfold_time = t.elapsed() - kfold_io_time;
        std::cout << "fold time: " << kfold_time << std::endl;
        results.setFloat("KFOLD_TIME", kfold_time);
        kfold_io_time = 0;

        std::cout << "graph -"
		  << " min: " << kfold->getMinGraph()->number_of_nodes()
		  << " maj: " << kfold->getMajGraph()->number_of_nodes() << std::endl;

        std::cout << "val -"
		  << " min: " << kfold->getMinValData()->size()
		  << " maj: " << kfold->getMajValData()->size() << std::endl;

	std::cout << "test -"
		  << " min: " << kfold->getMinTestData()->size()
		  << " maj: " << kfold->getMajTestData()->size() << std::endl;


	//flann build
        t.restart();

	auto minGraph = kfold->getMinGraph();
	auto majGraph = kfold->getMajGraph();
	auto minSize = minGraph->number_of_nodes();
	auto majSize = majGraph->number_of_nodes();
	auto size = minSize + majSize;
	auto features = kfold->getMinGraph()->getFeatureVec(0).size();


        // copy data to a linear vector, because flann needs a pointer
        FeatureVec data;
        data.reserve(size*features);

	forall_nodes((*minGraph),n) { 
                data.insert(data.end(), minGraph->getFeatureVec(n).begin(), minGraph->getFeatureVec(n).end());
	endfor }
	forall_nodes((*majGraph),n) { 
                data.insert(data.end(), majGraph->getFeatureVec(n).begin(), majGraph->getFeatureVec(n).end());
	endfor }

	std::cout << "size " << data.size() << std::endl;

        flann::Matrix<FeatureData> mat(data.data(), size, features);
        flann::Index<flann::L2<FeatureData>> index(mat, flann::KDTreeIndexParams(1));
        index.buildIndex();

	auto train_time = t.elapsed();
        std::cout << "train time: " << train_time << std::endl;
        results.setFloat("\tTRAIN_TIME", train_time);


	//flann predict
	size_t tp = 0, tn = 0, fp = 0, fn = 0;
        flann::SearchParams params(64);
        params.cores = partition_config.n_cores;
        params.cores = 1;

	std::cout << "min" << std::endl;

        FeatureVec minQuery;
	svm_data* minTestData = kfold->getMinTestData();
        minQuery.reserve(minTestData->size()*features);
        for (size_t i = 0; i < minTestData->size(); ++i) {
		FeatureVec tmp = svm_convert::node_to_feature((*minTestData)[i]);
                minQuery.insert(minQuery.end(), tmp.begin(), tmp.end());
        }
        flann::Matrix<FeatureData> minMat(minQuery.data(), size, features);
        std::vector<std::vector<int>> minIndices;
        std::vector<std::vector<FeatureData>> minDistances;
        index.knnSearch(minMat, minIndices, minDistances,
			partition_config.num_nn + 1, params);
	for (auto&& result : minIndices) {
		int count = 0;
		for (auto&& res : result) {
			if (res < minSize) {
				count++;
			}
		}

		if (count >= 5) {
			tp++;
		} else {
                        fn++;
                }
        }

	std::cout << "maj" << std::endl;


        FeatureVec majQuery;
	auto majTestData = kfold->getMajTestData();
        majQuery.reserve(majTestData->size()*features);
        for (size_t i = 0; i < majTestData->size(); ++i) {
		FeatureVec tmp = svm_convert::node_to_feature((*majTestData)[i]);
                majQuery.insert(majQuery.end(), tmp.begin(), tmp.end());
        }
        std::vector<std::vector<int>> majIndices;
        std::vector<std::vector<FeatureData>> majDistances;
        flann::Matrix<FeatureData> majMat(majQuery.data(), size, features);
        index.knnSearch(majMat, majIndices, majDistances,
			partition_config.num_nn + 1, params);
	for (auto&& result : majIndices) {
		int count = 0;
		for (auto&& res : result) {
			if (res < minSize) {
				count++;
			}
		}

		if (count < 5) {
			tn++;
		} else {
                        fp++;
                }
        }

        auto sens = (double)tp / (tp+fn);
        auto spec = (double)tn / (tn+fp);
	auto gmean = std::sqrt(sens * spec);
	auto acc = (double)(tp+tn) / (tp+tn+fp+fn);
        std::cout << std::setprecision(3)
                  << std::fixed
                  << " AC:" << acc
                  << " SN:" << sens
                  << " SP:" << spec
                  << " GM:" << gmean
		  << " TP:" << tp
		  << " TN:" << tn
		  << " FP:" << fp
		  << " FN:" << fn
		  << std::endl;

        results.setFloat("AC_TEST", acc);
        results.setFloat("SN_TEST", sens);
        results.setFloat("SP_TEST", spec);
        results.setFloat("GM_TEST", gmean);


        auto time_all = t_all.elapsed();

        auto time_iteration = time_all - kfold_io_time;

        std::cout << "iteration time: " << std::setprecision(4) << std::fixed
		  << time_iteration << std::endl;

        results.setFloat("TIME", time_iteration);

        if (partition_config.timeout != 0 && time_iteration > partition_config.timeout) {
                std::cout << "timeout reached exiting..." << std::endl;
                exit(123);
        }

        t_all.restart();
        t.restart();
	}
	}

	results.print();

	return 0;
}
