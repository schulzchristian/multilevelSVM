#ifndef LOW_DIAMETER_CLUSTERING_H
#define LOW_DIAMETER_CLUSTERING_H

#include "definitions.h"
#include "partition/coarsening/matching/matching.h"

#include <vector>
#include <unordered_set>

class low_diameter_clustering : public matching {
public:
        low_diameter_clustering();
        virtual ~low_diameter_clustering();

        void match(const PartitionConfig & config,
                   graph_access & G,
                   Matching & _matching,
                   CoarseMapping & coarse_mapping,
                   NodeID & no_of_coarse_vertices,
                   NodePermutationMap & permutation);

private:
        void remap_cluster_ids(const graph_access & G,
                               CoarseMapping & coarse_mapping,
                               NodeID & no_of_coarse_vertices,
			       const std::vector<std::pair<EdgeWeight,NodeID>> & C);
};

#endif /* LOW_DIAMETER_CLUSTERING_H */
