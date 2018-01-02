#ifndef SVM_CONVERT_H
#define SVM_CONVERT_H

#include <vector>
#include <svm.h>
#include "data_structure/graph_access.h"
#include "svm_definitions.h"

class svm_convert {
public:
        static const FeatureData EPS = 0.000001f;

        static svm_data gaccess_to_nodes(const graph_access & G);

        static svm_feature feature_to_node(const FeatureVec & vec);

        template<typename T>
        static std::vector<T> take_sample(const std::vector<T> & data, float percentage);

        static svm_data sample_from_graph(const graph_access & G, float amount);
};

template<typename T>
std::vector<T> svm_convert::take_sample(const std::vector<T> & data, float percentage) {
        std::vector<T> sample;
        sample.reserve(data.size() * percentage);

        for (auto&& entry : data) {
                float r = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);

                if (r > percentage) {
                        continue;
                }

                sample.push_back(entry);
        }

        return sample;
}

#endif /* SVM_CONVERT_H */
