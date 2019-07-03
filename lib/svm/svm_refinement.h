#ifndef REFINEMENT_H
#define REFINEMENT_H

#include "definitions.h"
#include "data_structure/graph_hierarchy.h"
#include "svm_result.h"
#include "partition/partition_config.h"

template<class T>
class svm_refinement
{
public:
        svm_refinement(graph_hierarchy & min_hierarchy, graph_hierarchy & maj_hierarchy,
                       const svm_result<T> & initial_result, PartitionConfig conf);

        virtual ~svm_refinement();

        bool is_done();
        int get_level();

        void uncoarse();

	svm_data uncoarse_graph(const graph_access & G,
				const CoarseMapping & coarse_mapping,
				const std::vector<NodeID> & sv);


	svm_data get_SV_neighbors(const graph_access & G,
                                  const CoarseMapping & coarse_mapping,
                                  const std::vector<NodeID> & sv);

        svm_data get_SV(const graph_access & G, const std::vector<NodeID> & sv);

        virtual svm_result<T> step(const svm_data & min_sample, const svm_data & maj_sample) = 0;

	graph_access * G_min;
	graph_access * G_maj;

protected:
        graph_hierarchy * min_hierarchy;
        graph_hierarchy * maj_hierarchy;
        svm_data uncoarsed_data_min;
        svm_data uncoarsed_data_maj;
        svm_result<T> result;

        bool training_inherit;
        int num_skip_ms;
};

#endif /* REFINEMENT_H */
