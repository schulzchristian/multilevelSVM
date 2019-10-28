#ifndef KFOLD_H
#define KFOLD_H

#include <vector>
#include <string>
#include <unordered_map>

#include "definitions.h"
#include "data_structure/graph_access.h"
#include "partition/partition_config.h"
#include "svm.h"
#include "svm/svm_definitions.h"


class k_fold
{
public:
        k_fold(PartitionConfig conf);
        virtual ~k_fold();

        bool next(double & io_time);

        graph_access* getMinGraph();
        graph_access* getMajGraph();
        svm_data* getMinValData();
        svm_data* getMajValData();
        svm_data* getMinTestData();
        svm_data* getMajTestData();

	int getIteration();

protected:
        /// do kfold stuff in here
        virtual void next_intern(double & io_time) = 0;

        int iterations;
        int cur_iteration;
	float validation_percent;
	bool validation_seperate;

        graph_access cur_min_graph;
        graph_access cur_maj_graph;

        svm_data cur_min_train;
        svm_data cur_maj_train;
        svm_data cur_min_val;
        svm_data cur_maj_val;
        svm_data cur_min_test;
        svm_data cur_maj_test;
};

#endif /* KFOLD_H */
