#include "k_fold.h"
#include "graph_io.h"
#include "timer.h"
#include "svm_flann.h"
#include "svm_convert.h"
#include "random_functions.h"

k_fold::k_fold(int num_iter) {
        this->iterations = num_iter;
        this->cur_iteration = -1;
}

k_fold::~k_fold() {
}


bool k_fold::next() {
        this->cur_iteration += 1;
        if (cur_iteration >= this->iterations) {
                std::cout << "-------------- K-FOLD DONE -------------- " << std::endl;
                return false;
        }

        std::cout << "------------- K-FOLD ITERATION " << this->cur_iteration
                  << " -------------" << std::endl;

        this->next_intern();

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
