#ifndef FIX_REFINEMENT_H
#define FIX_REFINEMENT_H

#include "svm/svm_refinement.h"
#include "svm/svm_solver.h"
#include "svm/svm_definitions.h"

template<class T>
class fix_refinement : public svm_refinement<T>
{
public:
	fix_refinement(graph_hierarchy & min_hierarchy, graph_hierarchy & maj_hierarchy,
		       const svm_result<T> & initial_result, PartitionConfig conf);
	virtual ~fix_refinement();

	svm_result<T> step(const svm_data & min_sample, const svm_data & maj_sample) override;

	static
	svm_result<T> train_fix(svm_solver<T> & solver,
				const svm_data & min_sample,
				const svm_data & maj_sample,
				float C, float gamma);

private:
	svm_param param;
};


#endif /* FIX_REFINEMENT_H */
