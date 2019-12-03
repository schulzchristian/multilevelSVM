#ifndef UD_REFINEMENT_H
#define UD_REFINEMENT_H

#include "definitions.h"
#include "data_structure/graph_hierarchy.h"
#include "svm_solver.h"
#include "svm_result.h"
#include "svm_refinement.h"

template<class T>
class ud_refinement : public svm_refinement<T>
{
public:
        ud_refinement(graph_hierarchy & min_hierarchy, graph_hierarchy & maj_hierarchy,
		      const svm_result<T> & initial_result, PartitionConfig conf);

	virtual ~ud_refinement();

        virtual svm_result<T> step(const svm_data & min_sample,
				   const svm_data & maj_sample);


	static svm_result<T> train_ud(svm_solver<T> & solver,
				      const svm_data & min_sample,
				      const svm_data & maj_sample);
private:

	svm_result<T> train_refinement(svm_solver<T> & solver,
				       const svm_data & min_sample,
				       const svm_data & maj_sample,
				       bool inherit_ud, float param_c, float param_g);

        bool inherit_ud;
};

#endif /* UD_REFINEMENT_H */
