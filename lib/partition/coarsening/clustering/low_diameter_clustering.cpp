#include "low_diameter_clustering.h"

#include <vector>
#include <unordered_map>
#include "tools/random_functions.h"

low_diameter_clustering::low_diameter_clustering() {
}

low_diameter_clustering::~low_diameter_clustering() {
}


void low_diameter_clustering::match(const PartitionConfig & config,
                                    graph_access & G,
                                    Matching & _matching,
                                    CoarseMapping & coarse_mapping,
                                    NodeID & no_of_coarse_vertices,
                                    NodePermutationMap & permutation) {
        permutation.resize(G.number_of_nodes());

        // assignment of values from the exponential distribution to vertices
        std::vector<double> delta(G.number_of_nodes());

        double beta = std::log(G.number_of_nodes()) / config.diameter_upperbound ;

        for (NodeID i = 0; i < G.number_of_nodes(); ++i) {
                delta[i] = random_functions::nextFromExp(beta);
        }

        // Decomp-Arb
        // coarse_mapping is C in paper
        coarse_mapping.resize(G.number_of_nodes(), std::numeric_limits<NodeID>::max());
        no_of_coarse_vertices = 0;

        std::vector<NodeID> frontier;
        std::vector<NodeID> nextFrontier;
        NodeID numVisited = 0;
        int rounds = 0;

        //TODO use set for detecting unvisited vertices
        // std::set unvisited;

        while (numVisited < G.number_of_nodes()) {
                // add unvisited nodes to frontier
                forall_nodes (G,v) {
                        if (coarse_mapping[v] == std::numeric_limits<NodeID>::max() && delta[v] < rounds + 1) {
                                frontier.push_back(v);
                                coarse_mapping[v] = v;
                        }
                } endfor
                numVisited += frontier.size();
                // parfor
                for (NodeID v : frontier) {
                        forall_out_edges (G, e, v) {
                                NodeID target = G.getEdgeTarget(e);
                                // CAS
                                if (coarse_mapping[target] == std::numeric_limits<NodeID>::max()) {
                                        coarse_mapping[target] = v;
                                        nextFrontier.push_back(target);
                                }
                        } endfor
                }
                frontier.swap(nextFrontier);
                nextFrontier.clear();
                rounds++;
        }

        remap_cluster_ids(G, coarse_mapping, no_of_coarse_vertices);
}

void low_diameter_clustering::remap_cluster_ids(const graph_access & G, CoarseMapping & coarse_mapping, NodeID & no_of_coarse_vertices) {
        no_of_coarse_vertices = 0;
        std::unordered_map<NodeID, NodeID> remap;
        forall_nodes(G, node) {
                NodeID cur_cluster = coarse_mapping[node];
                if (remap.find(cur_cluster) == remap.end()) {
                        remap[cur_cluster] = no_of_coarse_vertices++;
                }

                coarse_mapping[node] = remap[cur_cluster];
        } endfor
}
