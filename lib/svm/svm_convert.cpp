#include "svm_convert.h"

std::vector<svm_node> svm_convert::feature_to_node(const FeatureVec & vec) {
        std::vector<svm_node> nodes;
        size_t features = vec.size();

        for (size_t i = 0; i < features; ++i) {
                if (std::abs(vec[i]) < EPS) // skip zero valued features
                        continue;
                svm_node n;
                n.index = i+1;
                n.value = vec[i];
                nodes.push_back(n);
        }

        svm_node n; // end node
        n.index = -1;
        n.value = 0;
        nodes.push_back(n);

        return nodes;
}

std::vector<std::vector<svm_node>> svm_convert::sample_from_graph(const graph_access & G, float amount) {
        std::vector<std::vector<svm_node>> nodes;

        forall_nodes(G, n) {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

                if (r > amount) {
                        continue;
                }

                std::vector<svm_node> line = svm_convert::feature_to_node(G.getFeatureVec(n));
                nodes.push_back(line);
        } endfor

                  return nodes;
}
