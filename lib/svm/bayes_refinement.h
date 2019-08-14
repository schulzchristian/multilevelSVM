#ifndef BAYES_REFINEMENT_H
#define BAYES_REFINEMENT_H

#include "definitions.h"
#include "data_structure/graph_hierarchy.h"
#include "svm/svm_definitions.h"
#include "svm/svm_result.h"
#include "svm/svm_refinement.h"
#include "svm/svm_solver.h"

template<class T>
class bayes_refinement : public svm_refinement<T>
{
public:
        bayes_refinement(graph_hierarchy & min_hierarchy, graph_hierarchy & maj_hierarchy,
			 const svm_result<T> & initial_result, PartitionConfig conf);

	virtual ~bayes_refinement();

        virtual svm_result<T> step(const svm_data & min_sample, const svm_data & maj_sample);


	static svm_result<T> train_bayes(svm_solver<T> & solver,
					 const svm_data & min_sample,
					 const svm_data & maj_sample,
					 long seed);
private:

	// svm_result<T> train_refinement(svm_solver<T> & solver,
	// 			       const svm_data & min_sample,
	// 			       const svm_data & maj_sample,
	// 			       float param_c, float param_g);

	long seed;
};

#endif /* BAYES_REFINEMENT_H */
