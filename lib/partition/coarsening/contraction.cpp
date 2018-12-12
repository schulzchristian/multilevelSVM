/******************************************************************************
 * contraction.cpp
 *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 *
 ******************************************************************************
 * Copyright (C) 2013-2015 Christian Schulz <christian.schulz@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "contraction.h"
#include "partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.h"
#include "tools/macros_assertions.h"

contraction::contraction() {

}

contraction::~contraction() {

}

// for documentation see technical reports of christian schulz
void contraction::contract(const PartitionConfig & partition_config,
                           graph_access & G,
                           graph_access & coarser,
                           const Matching & edge_matching,
                           const CoarseMapping & coarse_mapping,
                           const NodeID & no_of_coarse_vertices,
                           const NodePermutationMap & permutation) const {

        if(partition_config.matching_type == CLUSTER_COARSENING
           || partition_config.matching_type == SIMPLE_CLUSTERING) {
                return contract_clustering(partition_config, G, coarser, edge_matching, coarse_mapping, no_of_coarse_vertices, permutation);
        }

        if(partition_config.combine) {
                coarser.resizeSecondPartitionIndex(no_of_coarse_vertices);
        }

        std::vector<NodeID> new_edge_targets(G.number_of_edges());
        forall_edges(G, e) {
                new_edge_targets[e] = coarse_mapping[G.getEdgeTarget(e)];
        } endfor

        std::vector<EdgeID> edge_positions(no_of_coarse_vertices, UNDEFINED_EDGE);

        //we dont know the number of edges jet, so we use the old number for
        //construction of the coarser graph and then resize the field according
        //to the number of edges we really got
        coarser.start_construction(no_of_coarse_vertices, G.number_of_edges());

        NodeID cur_no_vertices = 0;

        forall_nodes(G, n) {
                NodeID node = permutation[n];
                //we look only at the coarser nodes
                if(coarse_mapping[node] != cur_no_vertices)
                        continue;

                NodeID coarseNode = coarser.new_node();
                coarser.setNodeWeight(coarseNode, G.getNodeWeight(node));
                coarser.setFeatureVec(coarseNode, G.getFeatureVec(node));

                if(partition_config.combine) {
                        coarser.setSecondPartitionIndex(coarseNode, G.getSecondPartitionIndex(node));
                }

                // do something with all outgoing edges (in auxillary graph)
                forall_out_edges(G, e, node) {
                        visit_edge(G, coarser, edge_positions, coarseNode, e, new_edge_targets);
                } endfor

                //this node was really matched
                NodeID matched_neighbor = edge_matching[node];
                if(node != matched_neighbor) {
                        NodeWeight node_weight = G.getNodeWeight(node);
                        NodeWeight neighbor_weight = G.getNodeWeight(matched_neighbor);

                        //update weight of coarser node
                        NodeWeight new_coarse_weight = node_weight + neighbor_weight;
                        coarser.setNodeWeight(coarseNode, new_coarse_weight);

                        //update feature vector weighted
                        FeatureVec v1 = G.getFeatureVec(node);
                        FeatureVec v2 = G.getFeatureVec(matched_neighbor);

                        FeatureVec new_feature_vec = combineFeatureVec(v1, node_weight, v2, neighbor_weight);

                        coarser.setFeatureVec(coarseNode, new_feature_vec);

                        forall_out_edges(G, e, matched_neighbor) {
                                visit_edge(G, coarser, edge_positions, coarseNode, e, new_edge_targets);
                        } endfor
                }
                forall_out_edges(coarser, e, coarseNode) {
                       edge_positions[coarser.getEdgeTarget(e)] = UNDEFINED_EDGE;
                } endfor

                cur_no_vertices++;
        } endfor

        ASSERT_RANGE_EQ(edge_positions, 0, edge_positions.size(), UNDEFINED_EDGE);
        ASSERT_EQ(no_of_coarse_vertices, cur_no_vertices);

        //this also resizes the edge fields ...
        coarser.finish_construction();
}

void contraction::contract_clustering(const PartitionConfig & partition_config,
                              graph_access & G,
                              graph_access & coarser,
                              const Matching & edge_matching,
                              const CoarseMapping & coarse_mapping,
                              const NodeID & no_of_coarse_vertices,
                              const NodePermutationMap & permutation) const {

        if(partition_config.combine) {
                coarser.resizeSecondPartitionIndex(no_of_coarse_vertices);
        }

        //save partition map -- important if the graph is allready partitioned
        std::vector< int > partition_map(G.number_of_nodes());
        int k = G.get_partition_count();
        forall_nodes(G, node) {
                partition_map[node] = G.getPartitionIndex(node);
                G.setPartitionIndex(node, coarse_mapping[node]);
        } endfor

        G.set_partition_count(no_of_coarse_vertices);

        complete_boundary bnd(&G);
        bnd.build();
        bnd.getUnderlyingQuotientGraph(coarser);

        G.set_partition_count(k);

        // variables for calculating the feature vec of the coarse nodes
        std::vector<NodeWeight> block_size(no_of_coarse_vertices);
        int num_features = G.getFeatureVec(0).size();
        std::vector<FeatureVec> combined_feature_vecs(no_of_coarse_vertices, FeatureVec(num_features, 0));

        forall_nodes(G, node) {
                NodeID coarsed_node = coarse_mapping[node];
                G.setPartitionIndex(node, partition_map[node]);
                coarser.setPartitionIndex(coarsed_node, G.getPartitionIndex(node));

                addWeightedToVec(combined_feature_vecs[coarsed_node],
                                 G.getFeatureVec(node),
                                 G.getNodeWeight(node));
                block_size[coarsed_node] += G.getNodeWeight(node);

                if(partition_config.combine) {
                        coarser.setSecondPartitionIndex(coarse_mapping[node], G.getSecondPartitionIndex(node));
                }

        } endfor

        forall_nodes(coarser, node) {
                divideVec(combined_feature_vecs[node], block_size[node]);
                coarser.setFeatureVec(node, combined_feature_vecs[node]);
        endfor }
}


// for documentation see technical reports of christian schulz
void contraction::contract_partitioned(const PartitionConfig & partition_config,
                                       graph_access & G,
                                       graph_access & coarser,
                                       const Matching & edge_matching,
                                       const CoarseMapping & coarse_mapping,
                                       const NodeID & no_of_coarse_vertices,
                                       const NodePermutationMap & permutation) const {

        if(partition_config.matching_type == CLUSTER_COARSENING) {
                return contract_clustering(partition_config, G, coarser, edge_matching, coarse_mapping, no_of_coarse_vertices, permutation);
        }


        std::vector<NodeID> new_edge_targets(G.number_of_edges());
        forall_edges(G, e) {
                new_edge_targets[e] = coarse_mapping[G.getEdgeTarget(e)];
        } endfor

        std::vector<EdgeID> edge_positions(no_of_coarse_vertices, UNDEFINED_EDGE);

        //we dont know the number of edges jet, so we use the old number for
        //construction of the coarser graph and then resize the field according
        //to the number of edges we really got
        coarser.set_partition_count(G.get_partition_count());
        coarser.start_construction(no_of_coarse_vertices, G.number_of_edges());

        if(partition_config.combine) {
                coarser.resizeSecondPartitionIndex(no_of_coarse_vertices);
        }

        NodeID cur_no_vertices = 0;

        PRINT(std::cout <<  "contracting a partitioned graph"  << std::endl;)
        forall_nodes(G, n) {
                NodeID node = permutation[n];
                //we look only at the coarser nodes
                if(coarse_mapping[node] != cur_no_vertices)
                        continue;

                NodeID coarseNode = coarser.new_node();
                coarser.setNodeWeight(coarseNode, G.getNodeWeight(node));
                coarser.setPartitionIndex(coarseNode, G.getPartitionIndex(node));

                if(partition_config.combine) {
                        coarser.setSecondPartitionIndex(coarseNode, G.getSecondPartitionIndex(node));
                }
                // do something with all outgoing edges (in auxillary graph)
                forall_out_edges(G, e, node) {
                                visit_edge(G, coarser, edge_positions, coarseNode, e, new_edge_targets);
                } endfor

                //this node was really matched
                NodeID matched_neighbor = edge_matching[node];
                if(node != matched_neighbor) {
                        //update weight of coarser node
                        NodeWeight new_coarse_weight = G.getNodeWeight(node) + G.getNodeWeight(matched_neighbor);
                        coarser.setNodeWeight(coarseNode, new_coarse_weight);

                        forall_out_edges(G, e, matched_neighbor) {
                                visit_edge(G, coarser, edge_positions, coarseNode, e, new_edge_targets);
                        } endfor
                }
                forall_out_edges(coarser, e, coarseNode) {
                       edge_positions[coarser.getEdgeTarget(e)] = UNDEFINED_EDGE;
                } endfor

                cur_no_vertices++;
        } endfor

        ASSERT_RANGE_EQ(edge_positions, 0, edge_positions.size(), UNDEFINED_EDGE);
        ASSERT_EQ(no_of_coarse_vertices, cur_no_vertices);

        //this also resizes the edge fields ...
        coarser.finish_construction();
}

FeatureVec contraction::combineFeatureVec(const FeatureVec & vec1, NodeWeight weight1,
                                          const FeatureVec & vec2, NodeWeight weight2) const {
        size_t features = vec1.size();
        FeatureVec combined_features(features);

        for (size_t i = 0; i < features; ++i) {
                combined_features[i] = (weight1 * vec1[i] + weight2 * vec2[i])
                        / ((float)(weight1 + weight2));
        }

        return combined_features;
}

void contraction::divideVec(FeatureVec & vec, NodeWeight weights) const {
        // TODO use map aka. std::for_each
        size_t features = vec.size();

        for (size_t i = 0; i < features; ++i) {
                vec[i] /= (float) weights;
        }
}

void contraction::addWeightedToVec(FeatureVec & vec, const FeatureVec & vecToAdd, NodeWeight weight) const {
        size_t features = vec.size();

        for (size_t i = 0; i < features; ++i) {
                vec[i] += vecToAdd[i] * weight;
        }
}
