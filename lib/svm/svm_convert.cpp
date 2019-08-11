#include "svm_convert.h"
#include <unordered_set>

svm_feature svm_convert::feature_to_node(const FeatureVec & vec) {
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

svm_data svm_convert::graph_to_nodes(const graph_access & G) {
        std::vector<std::vector<svm_node>> nodes;

        forall_nodes(G, n) {
                nodes.push_back(svm_convert::feature_to_node(G.getFeatureVec(n)));
        } endfor

        return nodes;
}

svm_data svm_convert::graph_part_to_nodes(const graph_access & G, const std::vector<NodeID> & sv) {
        svm_data nodes;

        std::unordered_set<NodeID> sv_set{sv.begin(), sv.end()};

        forall_nodes(G, node) {
                if (sv_set.find(node) != sv_set.end()) {
                        nodes.push_back(svm_convert::feature_to_node(G.getFeatureVec(node)));
                }
        } endfor

        return nodes;
}


DataSet::node2d svm_convert::svmdata_to_dataset(const svm_data & data) {
	DataSet::node2d result(data.size());
	for (size_t i = 0; i < data.size(); i++) {
		result[i].reserve(data[i].size());
		for (size_t j = 0; j < data[i].size()-1; j++) {
			svm_node cur = data[i][j];
			DataSet::node node(cur.index, cur.value);
			result[i].push_back(node);
		}
	}
	return result;
}
