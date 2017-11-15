#ifndef SVM_CONVERT_H
#define SVM_CONVERT_H

#include <vector>
#include <svm.h>
#include "data_structure/graph_access.h"

class svm_convert {
public:
        static const FeatureData EPS = 0.000001f;

        static std::vector<std::vector<svm_node>> gaccess_to_nodes(const graph_access & G);

        static std::vector<svm_node> feature_to_node(const FeatureVec & vec);

        static std::vector<std::vector<svm_node>> svm_convert::convert_sample_to_nodes(const graph_access & G, float amount);
};

#endif /* SVM_CONVERT_H */
