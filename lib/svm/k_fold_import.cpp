#include "k_fold_import.h"
#include "graph_io.h"
#include "svm_io.h"
#include "svm_flann.h"
#include "random_functions.h"
#include "timer.h"


k_fold_import::k_fold_import(int num_exp, int num_iter, const std::string & basename)
        : k_fold(num_iter) {
        this->num_exp = num_exp;
        this->basename = basename;
}

k_fold_import::~k_fold_import() {
}

void k_fold_import::next_intern(double & io_time) {
        // TODO don't build the path with hard coded strings, use a path template as argument
        std::string min_train_name = this->basename + "kfold_p_train_data_exp_" + std::to_string(this->num_exp) + "_fold_" + std::to_string(k_fold::cur_iteration) + "_exp_0.1_data";
        std::string maj_train_name = this->basename + "kfold_n_train_data_exp_" + std::to_string(this->num_exp) + "_fold_" + std::to_string(k_fold::cur_iteration) + "_exp_0.1_data";
        std::string test_name = this->basename + "kfold_test_data_exp_" + std::to_string(this->num_exp) + "_fold_" + std::to_string(k_fold::cur_iteration) + "_exp_0.1_data";


        std::cout << "reading " << min_train_name << std::endl;

        int nn = 10;

        timer t;

        std::vector<FeatureVec> min_features;
        std::vector<std::vector<Edge>> min_edges;
        svm_io::readFeaturesLines(min_train_name, min_features);
        io_time += t.elapsed();
        svm_flann::run_flann(min_features, min_edges);
        graph_io::readGraphFromVec(this->cur_min_graph, min_edges, min_features.size() * nn * 2);
        graph_io::readFeatures(this->cur_min_graph, min_features);


        t.restart();
        std::vector<FeatureVec> maj_features;
        std::vector<std::vector<Edge>> maj_edges;
        svm_io::readFeaturesLines(maj_train_name, maj_features);
        io_time += t.elapsed();
        svm_flann::run_flann(maj_features, maj_edges);
        graph_io::readGraphFromVec(this->cur_maj_graph, maj_edges, maj_features.size() * nn * 2);
        graph_io::readFeatures(this->cur_maj_graph, maj_features);

        t.restart();
        svm_io::readTestSplit(test_name, this->cur_min_test, this->cur_maj_test);
        io_time += t.elapsed();
}
