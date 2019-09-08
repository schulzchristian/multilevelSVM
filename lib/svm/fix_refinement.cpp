#include "fix_refinement.h"
#include "svm_summary.h"
#include "svm_solver_factory.h"

template<class T>
fix_refinement<T>::fix_refinement(graph_hierarchy & min_hierarchy,
				  graph_hierarchy & maj_hierarchy,
				  const svm_result<T> & initial_result,
				  PartitionConfig conf)
	: svm_refinement<T>(min_hierarchy, maj_hierarchy, initial_result, conf) {
	this->param = std::make_pair(conf.fix_C, conf.fix_gamma);
}

template<class T>
fix_refinement<T>::~fix_refinement() {}

template<class T>
svm_result<T> fix_refinement<T>::step(const svm_data & min_sample, const svm_data & maj_sample) {
	std::cout << "FIX refinement at level " << this->get_level() << std::endl;

	this->uncoarse();

        std::cout << "current level nodes"
                  << " min " << this->uncoarsed_data_min.size()
                  << " maj " << this->uncoarsed_data_maj.size()
                  << std::endl;

        svm_instance instance;
        instance.read_problem(this->uncoarsed_data_min, this->uncoarsed_data_maj);
	std::unique_ptr<svm_solver<T>> solver = svm_solver_factory::create<T>(instance);

	svm_summary<T> summary = solver->train_single(this->param, min_sample, maj_sample);
	svm_result<T> res(std::vector<svm_summary<T>> {summary}, instance);
	return res;
}

template<class T>
svm_result<T> fix_refinement<T>::train_fix(svm_solver<T> & solver,
					   const svm_data & min_sample,
					   const svm_data & maj_sample,
					   float C,
					   float gamma) {
	svm_summary<T> summary = solver.train_single(std::make_pair(C, gamma),
						     min_sample, maj_sample);
	svm_result<T> res(std::vector<svm_summary<T>>{summary}, solver.get_instance());
	return res;
}

template class fix_refinement<svm_model>;
template class fix_refinement<SVC>;
