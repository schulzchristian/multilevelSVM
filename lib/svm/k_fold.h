#ifndef KFOLD_H
#define KFOLD_H

#include <vector>
#include <string>
#include <unordered_map>

#include "definitions.h"
#include "data_structure/graph_access.h"
#include "svm.h"


class k_fold
{
public:
        k_fold(int num_iter, const std::string & filename);
        virtual ~k_fold();

        bool next();

        graph_access* getMinGraph();
        graph_access* getMajGraph();
        std::vector<std::vector<svm_node>>* getMinTestData();
        std::vector<std::vector<svm_node>>* getMajTestData();

private:
        void readData(const std::string & filename);
        void calculate_kfold_class(const std::vector<FeatureVec> & features_full,
                                   graph_access & target_graph,
                                   std::vector<std::vector<svm_node>> & target_test);

        int iterations;
        int cur_iteration;

        std::vector<FeatureVec> min_features;
        std::vector<FeatureVec> maj_features;
        graph_access cur_min_graph;
        graph_access cur_maj_graph;
        std::vector<std::vector<svm_node>> cur_min_test;
        std::vector<std::vector<svm_node>> cur_maj_test;
};

#endif /* KFOLD_H */
