#include "k_fold_build.h"
#include "timer.h"
#include "graph_io.h"
#include "svm_io.h"
#include "svm_flann.h"
#include "svm_convert.h"
#include "random_functions.h"


k_fold_build::k_fold_build(int num_nn, int num_iter, const std::string & filename)
        : k_fold(num_iter) {
        this->num_nn = num_nn;
        readData(filename);
}

k_fold_build::~k_fold_build() {
}

void k_fold_build::readData(const std::string & filename) {
        timer t;

        svm_io::readFeaturesLines(filename + "_min_data", this->min_features);
        svm_io::readFeaturesLines(filename + "_maj_data", this->maj_features);

        random_functions::permutate_vector_good(this->min_features, false);
        random_functions::permutate_vector_good(this->maj_features, false);

        std::cout << "io time: " << t.elapsed() << std::endl;

        std::cout << "full graph -"
                  << " min: " << this->min_features.size()
                  << " maj: " << this->maj_features.size()
                  << " features: " << this->min_features[0].size() << std::endl;
}


void k_fold_build::next_intern(double & io_time) {
        this->cur_min_test.clear();
        this->cur_maj_test.clear();

        calculate_kfold_class(this->min_features, this->cur_min_graph, this->cur_min_test);
        calculate_kfold_class(this->maj_features, this->cur_maj_graph, this->cur_maj_test);
}

void k_fold_build::calculate_kfold_class(const std::vector<FeatureVec> & features_full,
                                   graph_access & target_graph,
                                   std::vector<std::vector<svm_node>> & target_test) {
        NodeID nodes          = features_full.size();
        NodeID remaining_part = nodes % this->iterations;
        NodeID test_size      = floor(nodes / this->iterations);
        NodeID test_start     = k_fold::cur_iteration * test_size;
        NodeID test_end       = (k_fold::cur_iteration + 1) * test_size;
        // NodeID train_size     = nodes - train_size;

        std::vector<FeatureVec> feature_subset(features_full);

        feature_subset.erase(feature_subset.begin() + test_start,
                                 feature_subset.begin() + test_end);

        std::vector<FeatureVec> test_subset = std::vector<FeatureVec>();
        test_subset.insert(test_subset.end(), features_full.begin() + test_start, features_full.begin() + test_end);

        std::vector<std::vector<Edge>> edges_subset;
        svm_flann::run_flann(feature_subset, edges_subset, this->num_nn);

        graph_io::readGraphFromVec(target_graph, edges_subset, nodes * this->num_nn * 2);
        graph_io::readFeatures(target_graph, feature_subset);

        target_test.reserve(test_size);
        for (const FeatureVec & f : test_subset) {
                target_test.push_back(svm_convert::feature_to_node(f));
        }
}
