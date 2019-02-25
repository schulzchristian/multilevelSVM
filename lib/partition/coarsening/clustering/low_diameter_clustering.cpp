#include "low_diameter_clustering.h"

#include <unordered_map>
#include <utility>
#include <algorithm>
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

        // first: values from the exponential distribution to vertices
	// second: the fractional part of its shift value
        std::vector<std::pair<double, double>> delta(G.number_of_nodes());

        // double beta = std::log(G.number_of_nodes()) / config.diameter_upperbound;
        double beta = config.diameter_upperbound;

        for (NodeID i = 0; i < G.number_of_nodes(); ++i) {
                delta[i].first = random_functions::nextFromExp(beta);
                delta[i].second = random_functions::next();
        }

        // Decomp-min
	std::vector<std::pair<EdgeWeight,NodeID>> C(G.number_of_nodes(), std::make_pair(std::numeric_limits<EdgeWeight>::max(),
											std::numeric_limits<NodeID>::max()));
	std::vector<NodeID> frontier;
        NodeID numVisited = 0;
        int rounds = 0;

	while (numVisited < G.number_of_nodes()) {
                // add new BFS centers
		forall_nodes(G, v) {
			// node is unvisited
			// and should be added in the current round
			if (C[v].first == std::numeric_limits<EdgeWeight>::max()
			    && delta[v].first < rounds + 1) {
                                frontier.push_back(v);
				C[v].first = -1;
				C[v].second = v;
                        }
                } endfor
                numVisited += frontier.size();
		std::unordered_set<NodeID> next_frontier;
		for (NodeID v : frontier) {
			forall_out_edges (G, e, v) {
                                NodeID w = G.getEdgeTarget(e);
				// EdgeWeight dist = C[v].first + G.getEdgeWeight(e);
				// if egde from frontier node to unvisited node
				if (C[w].first != -1
				    && C[w].first > delta[C[v].second].second) {
				// and frontier node is the nearest of the possible canditats this round
				    // && C[w].first > dist) {
				    // 	C[w].first = dist;
				        C[w].first = delta[C[v].second].second;
					C[w].second = C[v].second;
                                        next_frontier.insert(w);
                                }
				// else intercomponent edge
				// ignored handled later by the framework
                        } endfor
                }
                for (NodeID w : next_frontier) {
			C[w].first = -1;
		}
		frontier.clear();
		frontier.reserve(next_frontier.size());
		frontier.insert(frontier.end(), next_frontier.begin(), next_frontier.end());
		rounds++;
        }

        remap_cluster_ids(G, coarse_mapping, no_of_coarse_vertices, C);
}

void low_diameter_clustering::remap_cluster_ids(const graph_access & G,
						CoarseMapping & coarse_mapping,
						NodeID & no_of_coarse_vertices,
						const std::vector<std::pair<EdgeWeight,NodeID>> & C) {
        coarse_mapping.resize(G.number_of_nodes());
        no_of_coarse_vertices = 0;
        std::unordered_map<NodeID, NodeID> remap;
        forall_nodes(G, node) {
                NodeID cur_cluster = C[node].second;
                if (remap.count(cur_cluster) == 0) {
                        remap[cur_cluster] = no_of_coarse_vertices;
			no_of_coarse_vertices += 1;
		}

                coarse_mapping[node] = remap[cur_cluster];
        } endfor
}
