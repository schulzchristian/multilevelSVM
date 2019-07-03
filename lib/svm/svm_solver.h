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

template<class T>
class svm_solver
{
public:
        svm_solver();
        svm_solver(const svm_instance & instance);

        virtual void train() = 0;
        svm_result<T> train_ud(const svm_data & min_sample, const svm_data & maj_sample);
        svm_result<T> train_grid(const svm_data & min_sample, const svm_data & maj_sample);
        svm_result<T> train_bayesopt(const svm_data & min_sample, const svm_data & maj_sample);
        svm_result<T> train_refinement(const svm_data & min_sample, const svm_data & maj_sample,
				       bool inherit_ud, float param_c, float param_g);
	svm_result<T> train_range(const std::vector<svm_param> & params,
				  const svm_data & min_sample,
				  const svm_data & maj_sample);

	virtual std::vector<int> predict_batch(const svm_data & data);
        virtual int predict(const std::vector<svm_node> & node) = 0;

	virtual std::pair<std::vector<NodeID>, std::vector<NodeID>> get_SV() = 0;

        svm_summary<T> build_summary(const svm_data & min, const svm_data & maj);

        void set_C(float C);
        void set_gamma(float gamma);
	virtual void set_model(std::shared_ptr<T> new_model);

protected:
        svm_result<T> make_result(const std::vector<svm_summary<T>> & vec);

        svm_parameter param;
        svm_instance instance;
	std::shared_ptr<T> model;
};

#endif /* SVM_SOLVER_H */
