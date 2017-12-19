#include "k_fold.h"
#include "graph_io.h"
#include "timer.h"
#include "svm_flann.h"
#include "svm_convert.h"

#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clockinclude <algorithm>

k_fold::k_fold(int num_iter, const std::string & filename) {
        readData(filename);

        this->iterations = num_iter;
        this->cur_iteration = -1;
}

k_fold::~k_fold() {
}

void k_fold::readData(const std::string & filename) {
        timer t;

        graph_io::readFeaturesLines(filename + "_min_data", this->min_features);
        graph_io::readFeaturesLines(filename + "_maj_data", this->maj_features);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        std::shuffle(this->min_features.begin(), this->min_features.end(), std::default_random_engine(seed));
        std::shuffle(this->maj_features.begin(), this->maj_features.end(), std::default_random_engine(seed));

        std::cout << "io time: " << t.elapsed() << std::endl;

        std::cout << "full graph -"
                  << " min: " << this->min_features.size()
                  << " maj: " << this->maj_features.size()
                  << " features: " << min_features[0].size() << std::endl;
}

bool k_fold::next() {
        this->cur_iteration += 1;
        if (cur_iteration >= this->iterations) {
                std::cout << "-------------- K-FOLD DONE -------------- " << std::endl;
                return false;
        }

        std::cout << "------------- K-FOLD ITERATION " << this->cur_iteration + 1
                  << " -------------" << std::endl;

        this->cur_min_test.clear();
        this->cur_maj_test.clear();

        calculate_kfold_class(this->min_features, this->cur_min_graph, this->cur_min_test);
        calculate_kfold_class(this->maj_features, this->cur_maj_graph, this->cur_maj_test);

        return true;
}

void k_fold::calculate_kfold_class(const std::vector<FeatureVec> & features_full,
                                   graph_access & target_graph,
                                   std::vector<std::vector<svm_node>> & target_test) {
        int nn = 10;
        NodeID nodes          = features_full.size();
        NodeID remaining_part = nodes % this->iterations;
        NodeID test_size      = floor(nodes / this->iterations);
        NodeID test_start     = this->cur_iteration * test_size;
        NodeID test_end       = (this->cur_iteration + 1) * test_size;
        // NodeID train_size     = nodes - train_size;

        std::vector<FeatureVec> feature_subset(features_full);

        feature_subset.erase(feature_subset.begin() + test_start,
                                 feature_subset.begin() + test_end);

        std::vector<FeatureVec> test_subset = std::vector<FeatureVec>();
        test_subset.insert(test_subset.end(), features_full.begin() + test_start, features_full.begin() + test_end);

        std::vector<std::vector<Edge>> edges_subset;
        svm_flann::run_flann(feature_subset, edges_subset);

        graph_io::readGraphFromVec(target_graph, edges_subset, nodes * nn * 2);
        graph_io::readFeatures(target_graph, feature_subset);

        target_test.reserve(test_size);
        for (const FeatureVec & f : test_subset) {
                target_test.push_back(svm_convert::feature_to_node(f));
        }
}

graph_access* k_fold::getMinGraph() {
        return &this->cur_min_graph;
}

graph_access* k_fold::getMajGraph() {
        return &this->cur_maj_graph;
}

std::vector<std::vector<svm_node>>* k_fold::getMinTestData() {
        return &this->cur_min_test;
}

std::vector<std::vector<svm_node>>* k_fold::getMajTestData() {
        return &this->cur_maj_test;
}

void k_fold::setResult(const std::string &tag, float result) {
        if (this->results.count(tag) <= 0) {
                this->results[tag] = std::vector<float>(this->iterations, 0);
                this->tag_order.push_back(tag);
        }
        this->results[tag][this->cur_iteration] = result;
}

void k_fold::printAverages() {
        for (const auto& tag : this->tag_order) {
                std::vector<float> & results = this->results[tag];
                float average = 0;
                for (const float res : results){
                        average += res;
                }
                if (this->cur_iteration >= this->iterations) {
                        average /= this->iterations;
                } else {
                        average /= this->cur_iteration + 1;
                }

                std::cout << tag << "\t" << average << std::endl;
        }
}
