#ifndef SVMIO_H_
#define SVMIO_H_

#include <vector>

#include "definitions.h"
#include "data_structure/graph_access.h"
#include "svm/svm_definitions.h"
#include "tools/random_functions.h"


class svm_io {
public:
        static void readFeaturesLines(const std::string & filename, std::vector<FeatureVec> & data);

        static void readTestSplit(const std::string & filename, std::vector<svm_feature> & min_test_data,
                                  std::vector<svm_feature> & maj_test_data);

        template<typename T>
        static std::vector<T> take_sample(const std::vector<T> & data, float percentage);

        static svm_data sample_from_graph(const graph_access & G, float amount);

};

template<typename T>
std::vector<T> svm_io::take_sample(const std::vector<T> & data, float percentage) {
        std::vector<T> sample;
        sample.reserve(data.size() * percentage);

        for (const auto& entry : data) {
                if (random_functions::next() > percentage) {
                        continue;
                }

                sample.push_back(entry);
        }

        return sample;
}

#endif /* SVMIO_H_ */
