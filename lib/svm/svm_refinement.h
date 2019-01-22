#ifndef REFINEMENT_H
#define REFINEMENT_H

#include "definitions.h"
#include "data_structure/graph_hierarchy.h"
#include "svm_solver.h"
#include "svm_result.h"

class svm_refinement
{
public:
        svm_refinement(graph_hierarchy & min_hierarchy, graph_hierarchy & maj_hierarchy,
                       const svm_result & initial_result, int num_skip_ms, int inherit_ud);

        virtual ~svm_refinement();

        bool is_done();
        int get_level();

        void uncoarse();

        svm_result step(const svm_data & min_sample, const svm_data & maj_sample);

        svm_data get_SV_neighbors(const graph_access & G,
                                  const CoarseMapping & coarse_mapping,
                                  const std::vector<NodeID> & sv);

        svm_data get_SV(const graph_access & G, const std::vector<NodeID> & sv);

	graph_access * G_min;
	graph_access * G_maj;

private:
        graph_hierarchy * min_hierarchy;
        graph_hierarchy * maj_hierarchy;
        svm_data neighbors_min;
        svm_data neighbors_maj;
        svm_result result;
        bool inherit_ud;
        bool training_inherit;
        int num_skip_ms;
};

#endif /* REFINEMENT_H */
