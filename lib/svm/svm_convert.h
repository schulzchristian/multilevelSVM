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

        template<typename T>
        static std::vector<T> take_sample(const std::vector<T> & data, float percentage);

};

template<typename T>
std::vector<T> svm_convert::take_sample(const std::vector<T> & data, float percentage) {
        std::vector<std::vector<svm_node>> nodes;
        nodes.reserve(data.size() * percentage);

        for (auto&& entry : data) {
                float r = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);

                if (r > percentage) {
                        continue;
                }

                nodes.push_back(entry);
        }

        return nodes;
}

#endif /* SVM_CONVERT_H */
