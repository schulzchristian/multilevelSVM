#ifndef SVM_CONVERT_H
#define SVM_CONVERT_H

#include <vector>
#include <svm.h>
#include <thundersvm/dataset.h>
#include "data_structure/graph_access.h"
#include "svm_definitions.h"

class svm_convert {
public:
        static constexpr FeatureData EPS = 0.000001f;

        static svm_feature feature_to_node(const FeatureVec & vec);

        static FeatureVec node_to_feature(const svm_feature & data);

        static svm_data graph_to_nodes(const graph_access & G);

        static svm_data graph_part_to_nodes(const graph_access & G, const std::vector<NodeID> & sv);

        static DataSet::node2d svmdata_to_dataset(const svm_data & data);
};

#endif /* SVM_CONVERT_H */
