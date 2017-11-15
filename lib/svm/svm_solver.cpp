#include <iostream>
#include <algorithm>

#include "svm_solver.h"
#include "svm_convert.h"
#include "grid_search.h"
#include "timer.h"


void print_null(const char *s) {}


svm_solver::svm_solver() {
        svm_set_print_string_function(&print_null);

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
}

svm_solver::svm_solver(const svm_solver & solver) {
        this->param = solver.param;
        this->prob = solver.prob;
        this->model = nullptr;
}

svm_solver::~svm_solver() {
        svm_free_and_destroy_model(&(this->model));
        // svm_destroy_param(&(this->param));
        // if (prob.y) {
        //         delete[] prob.y;
        // }
        // if (prob.x) {
        //         for (int i = 0; i < this->prob.l; ++i) {
        //                 delete [] this->prob.x[i];
        //         }
        // }
}

void svm_solver::read_problem(const graph_access & G_maj, const graph_access & G_min) {
        size_t features = G_maj.getFeatureVec(0).size();

        this->param.gamma = 1/(float) features;

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
}

void svm_solver::add_graph_to_problem(const graph_access & G, int label, NodeID offset) {
        const FeatureData eps = 0.000001;
        size_t features = G.getFeatureVec(0).size();

        forall_nodes(G, node) {
                NodeID prob_node = node + offset;
                this->prob.y[prob_node] = label;

                const FeatureVec vec = G.getFeatureVec(node);
                int att_num = 0;
                for (size_t i = 0; i < features; ++i) {
                        if (abs(vec[i]) > eps) // skip zero valued features
                                continue;

                        svm_node n;
                        n.index = i+1;
                        n.value = vec[i];
                        this->prob.x[prob_node][att_num] = n;
                        ++att_num;
                }
                svm_node n; // end node
                n.index = -1;
                n.value = 0;
                this->prob.x[prob_node][att_num] = n;
        } endfor
}

void svm_solver::train() {
        const char * error_msg = svm_check_parameter(&(this->prob), &(this->param));
        if (error_msg != NULL) {
                std::cout << error_msg << std::endl;
        }

        this->model = svm_train(&(this->prob), &(this->param));
}

void svm_solver::train_initial(const std::vector<std::vector<svm_node>>& maj_sample,
                               const std::vector<std::vector<svm_node>>& min_sample) {
        const char * error_msg = svm_check_parameter(&(this->prob), &(this->param));
        if (error_msg != NULL) {
                std::cout << error_msg << std::endl;
        }


        std::vector<std::pair<svm_solver,svm_summary>> models;

        double training_time = 0;
        double validation_time = 0;

        //first grid search
        grid_search gs(-5,15,2,3,-15,-2);
        auto params = gs.get_sequence();

        // could be done in parallel
        for (auto&& p : params) {
                svm_solver cur_solver(*this);
                cur_solver.param.C = pow(2, p.first);
                cur_solver.param.gamma = pow(2, p.second);

                timer t;
                cur_solver.train();
                training_time += t.elapsed();

                t.restart();
                svm_summary cur_summary = cur_solver.predict_validation_data(maj_sample, min_sample);
                cur_summary.C_log = p.first;
                cur_summary.gamma_log = p.second;

                validation_time += t.elapsed();

                models.push_back(std::make_pair(cur_solver, cur_summary));
        }

        svm_summary good = svm_solver::select_best_model(models);

        std::cout << "GOOD log best C=" << good.C_log << " log gamma=" << good.gamma_log << std::endl;
        good.print();

        //second (finer) grid search
        grid_search gs2 = grid_search::around(good.C_log, 2, 0.25, good.gamma_log, 2, 0.25);
        params = gs2.get_sequence();
        for (auto&& p : params) {
                if (abs(p.first - good.C) < svm_convert::EPS && abs(p.second - good.gamma) < svm_convert::EPS)
                        continue; // skip the already processed instance

                svm_solver cur_solver(*this);
                cur_solver.param.C = pow(2, p.first);
                cur_solver.param.gamma = pow(2, p.second);

                timer t;
                cur_solver.train();
                training_time += t.elapsed();

                t.restart();
                svm_summary cur_summary = cur_solver.predict_validation_data(maj_sample, min_sample);
                cur_summary.C_log = p.first;
                cur_summary.gamma_log = p.second;

                validation_time += t.elapsed();

                models.push_back(std::make_pair(cur_solver, cur_summary));
        }

        svm_summary best = svm_solver::select_best_model(models);

        std::cout << "trained and validated " << params.size() << " parameter combinations." << std::endl;
        std::cout << "trainig time: " << training_time << " validation time: " << validation_time << std::endl;
        std::cout << "BEST log C=" << best.C_log << " log gamma=" << best.gamma_log << std::endl;
        this->param.C = best.C;
        this->param.gamma = best.gamma;
        best.print();

        this->train();
}

svm_summary svm_solver::select_best_model(std::vector<std::pair<svm_solver,svm_summary>> & vec) {
        std::sort(vec.begin(), vec.end(), [](const std::pair<svm_solver,svm_summary> & a,
                                             const std::pair<svm_solver,svm_summary> & b){
                          return summary_cmp_better_gmean_sn::comp(a.second, b.second);
                  });


        for (size_t i = 0; i < vec.size(); ++i){
                if(vec[i].second.Gmean > 0.05)  // ignore the gmean zero
                        return vec[i].second;

        }
        return vec[0].second;   // in case there is model with gmean larger than zero, return the 1st one
}

int svm_solver::predict(const std::vector<svm_node> & nodes) {
        return svm_predict(this->model, nodes.data());
}

svm_summary svm_solver::predict_validation_data(const std::vector<std::vector<svm_node>> & maj,
                                                const std::vector<std::vector<svm_node>> & min) {
        size_t tp = 0, tn = 0, fp = 0, fn = 0;
        for (const auto& instance : maj) {
                int res = this->predict(instance);
                if (res == -1) {
                        tn++;
                } else {
                        fp++;
                }
        }

        for (const auto& instance : min) {
                int res = this->predict(instance);
                if (res == 1) {
                        tp++;
                } else {
                        fn++;
                }
        }

        svm_summary summary(*(this->model), tp, tn, fp, fn);

        return summary;
}
