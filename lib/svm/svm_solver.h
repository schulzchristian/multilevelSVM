#ifndef SVM_SOLVER_H
#define SVM_SOLVER_H

#include <vector>
#include <utility>
#include <memory>
#include <svm.h>

#include "svm_definitions.h"
#include "data_structure/graph_access.h"
#include "svm_summary.h"
#include "svm_desc.h"

class svm_solver
{
public:
        svm_solver();

        void read_problem(const svm_data & min_data, const svm_data & maj_data);
        void read_problem(const graph_access & G_min, const graph_access & G_maj);

        void train();
        svm_result train_initial(const svm_data & min_sample, const svm_data & maj_sample);
        svm_result train_range(const std::vector<svm_param> & params,
                               const svm_data & min_sample,
                               const svm_data & maj_sample);

        int predict(const std::vector<svm_node> & node);

        svm_summary predict_validation_data(const svm_data & min, const svm_data & maj);

        void dont_delete();
        void set_C(float C);
        void set_gamma(float gamma);

private:
        void allocate_prob(NodeID total_size, size_t features);
        void add_to_problem(const svm_data & data, int label, NodeID offset);
        void add_graph_to_problem(const graph_access & G, int label, NodeID offset);

        static svm_summary select_best_model(std::vector<svm_summary> & vec);
        static svm_result make_result(const std::vector<svm_summary> & vec);

        svm_desc desc;
        std::shared_ptr<std::vector<double>> prob_labels;
        std::shared_ptr<svm_data> prob_nodes;
        std::shared_ptr<std::vector<svm_node*>> prob_nodes_meta;
        svm_parameter param;
        std::shared_ptr<svm_model> model;

};

#endif /* SVM_SOLVER_H */
