#include "svm_convert.h"

std::vector<std::vector<svm_node>> svm_convert::gaccess_to_nodes(const graph_access & G) {
        std::vector<std::vector<svm_node>> nodes;

        forall_nodes(G, n) {
                std::vector<svm_node> line = svm_convert::feature_to_node(G.getFeatureVec(n));
                nodes.push_back(line);
        } endfor

        return nodes;
}


std::vector<svm_node> svm_convert::feature_to_node(const FeatureVec & vec) {
        std::vector<svm_node> nodes;
        size_t features = vec.size();

        for (size_t i = 0; i < features; ++i) {
                if (abs(vec[i]) > EPS) // skip zero valued features
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

std::vector<std::vector<svm_node>> svm_convert::convert_sample_to_nodes(const graph_access & G, float amount) {
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

std::vector<std::vector<svm_node>> svm_convert::convert_sample_to_nodes_int(const graph_access & G, NodeID num_nodes) {
        std::vector<std::vector<svm_node>> nodes;

        float percentage = num_nodes / (float) G.number_of_nodes();

        forall_nodes(G, n) {
                float r = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);

                if (r > percentage) {
                        continue;
                }

                std::vector<svm_node> line = svm_convert::feature_to_node(G.getFeatureVec(n));
                nodes.push_back(line);
        } endfor

        return nodes;
}
