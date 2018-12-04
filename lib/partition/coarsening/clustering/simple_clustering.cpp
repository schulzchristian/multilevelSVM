#include "simple_clustering.h"

#include <utility>

#include "algorithms/jarnik_prim.h"
#include "data_structure/graph_access.h"


simple_clustering::simple_clustering() {}

simple_clustering::~simple_clustering() {}

void simple_clustering::match(const PartitionConfig & config,
                              graph_access & G,
                              Matching & _matching,
                              CoarseMapping & coarse_mapping,
                              NodeID & no_of_coarse_vertices,
                              NodePermutationMap & permutation) {
        permutation.resize(G.number_of_nodes());
        coarse_mapping.resize(G.number_of_nodes(), std::numeric_limits<NodeID>::max());

        auto tree_pair = jarnik_prim::spanning_tree(G);
        graph_access* tree = tree_pair.first;
        NodeID root = tree_pair.second;

        this->cur_cluster = 0;
        this->coarse_mapping = &coarse_mapping;
        this->tree = tree;
        this->max_cluster_nodes = config.cluster_upperbound;

        visit_children(root);

        // unvisited nodes are single coarse nodes
        for (auto & coarseID : coarse_mapping) {
                if (coarseID == std::numeric_limits<NodeID>::max()) {
                        coarseID = ++cur_cluster;
                }
        }

        no_of_coarse_vertices = cur_cluster + 1;

        delete tree;
}

// we have a tree and it is assured that we are not visiting a node twice
void simple_clustering::visit_children(NodeID cur_node) {
        if (cur_cluster_nodes >= max_cluster_nodes) {
                cur_cluster++;
                cur_cluster_nodes = 0;
        }
        (*coarse_mapping)[cur_node] = cur_cluster;
        cur_cluster_nodes++;

        forall_out_edges ((*tree), e, cur_node) {
                visit_children(tree->getEdgeTarget(e));
        } endfor
}
