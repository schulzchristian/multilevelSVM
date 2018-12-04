#include "jarnik_prim.h"

#include <queue>
#include <utility>

#include "definitions.h"
#include "data_structure/graph_access.h"
#include "tools/random_functions.h"


struct _Edge {
        EdgeID id;
        NodeID from;
        NodeID to;
        EdgeWeight weight;

        _Edge(EdgeID i, NodeID f, NodeID t, NodeID w) : id(i), from(f), to(t), weight(w) {
        }
};

bool operator<(const _Edge& lhs, const _Edge& rhs) {
        return lhs.weight < rhs.weight;
}

std::pair<graph_access*,NodeID> jarnik_prim::spanning_tree(const graph_access & G) {
        NodeID size = G.number_of_nodes();

        // could be faster with priority queue of Nodes but this needs a decreaseKey operation
        // and algorithms in data_struture don't support arbitrary additional data
        std::priority_queue<_Edge> pq;

        std::vector<NodeID> parent = std::vector<NodeID>(size, std::numeric_limits<NodeID>::max());

        NodeID start_id = random_functions::nextInt(0, size-1);

        // put start_id egdes in priority queue
        forall_out_edges (G, e, start_id) {
                _Edge edge = _Edge(e, start_id, G.getEdgeTarget(e), G.getEdgeWeight(e));
                pq.push(std::move(edge));
        } endfor

        parent[start_id] = start_id;

        while (!pq.empty()) {
                _Edge cur_edge = pq.top();
                pq.pop();
                NodeID cur_node = cur_edge.to;
                // add node only if not already in the spanning tree
                if (parent[cur_node] != std::numeric_limits<NodeID>::max()) {
                        continue;
                }

                forall_out_edges (G, e, cur_node) {
                        NodeID target = G.getEdgeTarget(e);
                        // add node only if not already in the spanning tree
                        if (parent[target] != std::numeric_limits<NodeID>::max()) {
                                continue;
                        }
                        _Edge edge = _Edge(e, cur_node, target, G.getEdgeWeight(e));
                        pq.push(std::move(edge));
                } endfor

                parent[cur_node] = cur_edge.from;
        }

        std::vector<std::vector<NodeID>> children = std::vector<std::vector<NodeID>>(size);

        int test = 0;

        for (size_t i = 0; i < size; ++i) {
                if (i == start_id) continue;
                if (parent[i] == std::numeric_limits<NodeID>::max()) {
                        test++;
                        continue;
                }
                children[parent[i]].size();
                children[parent[i]].push_back(i);
        }

        graph_access* tree = new graph_access();
        tree->start_construction(size, size);

        for (size_t i = 0; i < size; ++i) {
                tree->new_node();

                for (auto c : children[i]) {
                        tree->new_edge(i, c);
                }
        }

        return std::make_pair(tree, start_id);
}
