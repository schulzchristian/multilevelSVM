#include "io/graph_io.h"
#include "svm/k_fold.h"
#include "svm/svm_flann.h"
#include "svm/svm_convert.h"
#include "tools/random_functions.h"
#include "tools/timer.h"

k_fold::k_fold(int num_iter) {
        this->iterations = num_iter;
        this->cur_iteration = -1;
}

k_fold::~k_fold() {
}


bool k_fold::next(double & io_time) {
        this->cur_iteration += 1;
        if (cur_iteration >= this->iterations) {
                std::cout << "-------------- K-FOLD DONE -------------- " << std::endl;
                return false;
        }

        std::cout << "------------- K-FOLD ITERATION " << this->cur_iteration
                  << " -------------" << std::endl;

        this->next_intern(io_time);

        return true;
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
