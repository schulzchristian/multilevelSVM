#include <iostream>
#include "svm_solver.h"

svm_solver::svm_solver() {
        // default values
        this->param.svm_type = C_SVC;
        this->param.kernel_type = RBF;
        this->param.degree = 3;
        this->param.gamma = 2;	// 1/num_features
        this->param.coef0 = 0;
        this->param.nu = 0.5;
        this->param.cache_size = 100;
        this->param.C = 32;
        this->param.eps = 1e-3;
        this->param.p = 0.1;
        this->param.shrinking = 1;
        this->param.probability = 0;
        this->param.nr_weight = 0;
        this->param.weight_label = NULL;
        this->param.weight = NULL;
        // cross_validation = 0;
}

svm_solver::~svm_solver() {
        // svm_free_and_destroy_model(&(this->model));
        // svm_destroy_param(&(this->param));
        if (prob.y) {
                delete[] prob.y;
        }
        if (prob.x) {
                for (int i = 0; i < this->prob.l; ++i) {
                        delete [] this->prob.x[i];
                }
        }
}

void svm_solver::read_problem(const graph_access & G_maj, const graph_access & G_min) {
        size_t features = G_maj.getFeatureVec(0).size();

        this->param.gamma = 1/(float) features;

        std::cout << "gamma " << param.gamma << "\n";

        this->prob.l = G_maj.number_of_nodes() + G_min.number_of_nodes();
        this->prob.y = new double [this->prob.l];
        this->prob.x = new svm_node* [this->prob.l];
        for (int i = 0; i < this->prob.l; ++i) {
                //this is probably bigger than needed because we omit zero valued entries
                this->prob.x[i] = new svm_node[features+1];
        }

        // vector<vector<svm_node> > nodes(prob.l, vector<svm_node>());

        add_graph_to_problem(G_maj, -1, 0);
        add_graph_to_problem(G_min, 1, G_maj.number_of_nodes());


        // for (int i = 0; i < 5; i++) {
        //         std::cout << this->prob.y[i] << " ";

        //         int j = 0;
        //         svm_node node = this->prob.x[i][j];
        //         while (node.index != -1) {
        //                 std::cout << node.index << "," << node.value << " ";
        //                 node = this->prob.x[i][++j];
        //         }
        //         std::cout << std::endl;
        // }
        // for (int i = G_maj.number_of_nodes(); i < G_maj.number_of_nodes()+5; i++) {
        //         std::cout << "label:" << this->prob.y[i] << " " << std::endl;

        //         int j = 0;
        //         svm_node node = this->prob.x[i][j];
        //         while (node.index != -1) {
        //                 std::cout << node.index << "," << node.value << " ";
        //                 node = this->prob.x[i][++j];
        //         }
        //         std::cout << std::endl;
        // }
}

void svm_solver::add_graph_to_problem(const graph_access & G, int label, NodeID offset) {
        const FeatureData eps = 0.000001;
        size_t features = G.getFeatureVec(0).size();

        forall_nodes(G, node) {
                NodeID prob_node = node + offset;
                this->prob.y[prob_node] = label;

                const FeatureVec vec = G.getFeatureVec(node);
                int att_num = 0;
                for (size_t i = 0; i < features; ++i, ++att_num) {
                        if (abs(vec[i]) > eps) // skip zero valued features
                                continue;

                        svm_node n;
                        n.index = i+1;
                        n.value = vec[i];
                        this->prob.x[prob_node][att_num] = n;
                }
                svm_node n; // end node
                n.index = -1;
                n.value = 0;
                this->prob.x[prob_node][att_num] = n;
        } endfor
}

void svm_solver::train() {
        char * error_msg = svm_check_parameter(&(this->prob), &(this->param));
        if (error_msg != NULL) {
                std::cout << error_msg << std::endl;
        }

        this->model = svm_train(&(this->prob), &(this->param));
}

int svm_solver::predict(const FeatureVec & vec) {
        size_t features = vec.size();

        std::vector<svm_node> nodes = std::vector<svm_node>(features);

        // svm_node node = new svm_node[features];
        for (size_t i = 0; i < features; ++i) {
                nodes[i].index = i+1;
                nodes[i].value = vec[i];
        }

        int res = svm_predict(this->model, nodes.data());

        return res;
}
