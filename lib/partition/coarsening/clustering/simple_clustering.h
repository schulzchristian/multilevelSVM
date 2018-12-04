#ifndef SIMPLE_CLUSTERING_H
#define SIMPLE_CLUSTERING_H

#include "definitions.h"
#include "partition/coarsening/matching/matching.h"


class simple_clustering : public matching {
public:
        simple_clustering();
        virtual ~simple_clustering();

        void match(const PartitionConfig & config,
                   graph_access & G,
                   Matching & _matching,
                   CoarseMapping & coarse_mapping,
                   NodeID & no_of_coarse_vertices,
                   NodePermutationMap & permutation);
private:
        void visit_children(NodeID cur_node);

        graph_access* tree;
        CoarseMapping* coarse_mapping;
        NodeID cur_cluster = 0;
        NodeID cur_cluster_nodes = 0;
        NodeID max_cluster_nodes;
};

#endif /* SIMPLE_CLUSTERING_H */
