#ifndef REFINEMENT_H
#define REFINEMENT_H

#include "definitions.h"
#include "data_structure/graph_hierarchy.h"
#include "svm_solver.h"

class svm_refinement
{
public:
        svm_refinement();
        virtual ~svm_refinement();

        svm_result main(graph_hierarchy & min_hierarchy, graph_hierarchy & maj_hierarchy,
                        svm_solver & final_solver, const svm_result & initial_result,
                        const svm_data & min_sample, const svm_data & maj_sample);

        svm_data get_SV_neighbors(const graph_access & G,
                                  const CoarseMapping & coarse_mapping,
                                  const std::vector<NodeID> & sv);
};

#endif /* REFINEMENT_H */
