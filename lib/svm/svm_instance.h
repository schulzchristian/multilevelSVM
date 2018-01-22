#ifndef SVM_INSTANCE_H
#define SVM_INSTANCE_H

#include <memory>

#include "definitions.h"
#include "data_structure/graph_access.h"
#include "svm_definitions.h"

class svm_instance
{
public:
        svm_instance();

        void read_problem(const svm_data & min_data, const svm_data & maj_data);
        void read_problem(const graph_access & G_min, const graph_access & G_maj);

        int size();
        double* label_data();
        svm_node** node_data();

        NodeID num_min;
        NodeID num_maj;
        NodeID features;

private:
        void allocate_prob(NodeID total_size, size_t features);
        void add_to_problem(const svm_data & data, int label);
        void add_to_problem(const graph_access & G, int label);

        std::shared_ptr<std::vector<double>> labels;
        std::shared_ptr<svm_data> nodes;
        std::shared_ptr<std::vector<svm_node*>> nodes_meta;
};


#endif /* SVM_INSTANCE_H */
