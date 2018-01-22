#include "svm_instance.h"
#include "svm_convert.h"

svm_instance::svm_instance() {
}

void svm_instance::read_problem(const svm_data & min_data, const svm_data & maj_data) {
        this->num_min = min_data.size();
        this->num_maj = maj_data.size();
        this->features = min_data[0].size();

        allocate_prob(min_data.size() + maj_data.size(), min_data[0].size());

        add_to_problem(min_data, 1);
        add_to_problem(maj_data, -1);
}

void svm_instance::read_problem(const graph_access & G_min, const graph_access & G_maj) {
        this->num_min = G_min.number_of_nodes();
        this->num_maj = G_maj.number_of_nodes();
        this->features = G_min.getFeatureVec(0).size();

        allocate_prob(G_min.number_of_nodes() + G_maj.number_of_nodes(), G_min.getFeatureVec(0).size());

        add_to_problem(G_min, 1);
        add_to_problem(G_maj, -1);
}

void svm_instance::allocate_prob(NodeID total_size, size_t features) {
        this->labels = std::make_shared<std::vector<double>>();
        this->nodes = std::make_shared<svm_data>();
        this->nodes_meta = std::make_shared<std::vector<svm_node*>>();

        this->labels->reserve(total_size);
        this->nodes->reserve(total_size);
        this->nodes_meta->reserve(total_size);
}

void svm_instance::add_to_problem(const svm_data & data, int label) {
        for (NodeID node = 0; node < data.size(); node++) {
                this->labels->push_back(label);

                this->nodes->push_back(data[node]);
                this->nodes_meta->push_back(this->nodes->back().data());
        }
}

void svm_instance::add_to_problem(const graph_access & G, int label) {
        forall_nodes(G, node) {
                this->labels->push_back(label);

                const FeatureVec vec = G.getFeatureVec(node);
                svm_feature svm_nodes = svm_convert::feature_to_node(vec);
                this->nodes->push_back(std::move(svm_nodes));
                this->nodes_meta->push_back(this->nodes->back().data());
        } endfor
}

int svm_instance::size() {
        return this->labels->size();
}

double* svm_instance::label_data() {
        return this->labels->data();
}

svm_node** svm_instance::node_data() {
        return this->nodes_meta->data();
}
