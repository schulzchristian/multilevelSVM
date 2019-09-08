#ifndef BAYES_REFINEMENT_H
#define BAYES_REFINEMENT_H

#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>
#include <bopt_state.hpp>
#include "svm/svm_definitions.h"
#include "svm/svm_refinement.h"
#include "svm/svm_solver.h"

template<class T>
class bayes_refinement : public svm_refinement<T>
{
public:
        bayes_refinement(graph_hierarchy & min_hierarchy, graph_hierarchy & maj_hierarchy,
			 const svm_result<T> & initial_result, PartitionConfig conf,
			 bayesopt::BOptState state);

	virtual ~bayes_refinement();

        svm_result<T> step(const svm_data & min_sample, const svm_data & maj_sample) override;


	static svm_result<T> train_bayes(svm_solver<T> & solver,
					 const svm_data & min_sample,
					 const svm_data & maj_sample,
					 bayesopt::BOptState & state,
					 int optimization_steps,
					 long seed);
private:
	long seed;
	int fix_num_vert_stop;
	int bayes_max_steps;
	bayesopt::BOptState opt_state;
};

#endif /* BAYES_REFINEMENT_H */
