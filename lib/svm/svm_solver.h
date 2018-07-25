#ifndef SVM_SOLVER_H
#define SVM_SOLVER_H

#include <vector>
#include <utility>
#include <memory>
#include <svm.h>

#include "svm_definitions.h"
#include "data_structure/graph_access.h"
#include "svm_summary.h"
#include "svm_result.h"

class svm_solver
{
public:
        svm_solver();
        svm_solver(const svm_instance & instance);

        void train();
        svm_result train_initial(const svm_data & min_sample, const svm_data & maj_sample);
        svm_result train_refinement(const svm_data & min_sample, const svm_data & maj_sample,
                                    bool inherit_ud, float param_c, float param_g);
        svm_result train_range(const std::vector<svm_param> & params,
                               const svm_data & min_sample,
                               const svm_data & maj_sample);

        int predict(const std::vector<svm_node> & node);

        svm_summary predict_validation_data(const svm_data & min, const svm_data & maj);

        void set_C(float C);
        void set_gamma(float gamma);

private:
        svm_result make_result(const std::vector<svm_summary> & vec);

        svm_parameter param;
        svm_instance instance;
        std::shared_ptr<svm_model> model;

};

#endif /* SVM_SOLVER_H */
