#include <iostream>
#include <unordered_set>

#include "svm_refinement.h"
#include "svm_instance.h"
#include "svm_convert.h"
#include "param_search.h"
#include "timer.h"

svm_refinement::svm_refinement(graph_hierarchy & min_hierarchy, graph_hierarchy & maj_hierarchy,
                               const svm_result & initial_result, int num_skip_ms, int inherit_ud)
    : result(initial_result) {
        this->min_hierarchy = &min_hierarchy;
        this->maj_hierarchy = &maj_hierarchy;
        this->neighbors_min = svm_convert::graph_to_nodes(* this->min_hierarchy->get_coarsest());
        this->neighbors_maj = svm_convert::graph_to_nodes(* this->maj_hierarchy->get_coarsest());
        this->training_inherit = false;
        this->num_skip_ms = num_skip_ms;
        this->inherit_ud = inherit_ud;
}

svm_refinement::~svm_refinement() {
}

bool svm_refinement::is_done() {
        return min_hierarchy->isEmpty() && maj_hierarchy->isEmpty();
}

int svm_refinement::get_level() {
        return std::max(min_hierarchy->size(), maj_hierarchy->size());
}

void svm_refinement::uncoarse() {
        std::vector<NodeID> sv_min = result.best().SV_min;
        std::vector<NodeID> sv_maj = result.best().SV_maj;

        // if maj_hierarchy is larger then start by only uncoarse the maj graph
        if (!min_hierarchy->isEmpty() && min_hierarchy->size() >= maj_hierarchy->size()) {
                std::cout << "minority uncoarsed" << std::endl;
                graph_access* G_min = min_hierarchy->pop_finer_and_project();
                CoarseMapping* mapping_min = min_hierarchy->get_mapping_of_current_finer();
                this->neighbors_min = get_SV_neighbors(*G_min, *mapping_min, sv_min);
                this->training_inherit = true; // after the first uncoarsening of the min data inherit params
        }
        if (!maj_hierarchy->isEmpty()) {
                std::cout << "majority uncoarsed" << std::endl;
                graph_access* G_maj = maj_hierarchy->pop_finer_and_project();
                CoarseMapping* mapping_maj = maj_hierarchy->get_mapping_of_current_finer();
                this->neighbors_maj = get_SV_neighbors(*G_maj, *mapping_maj, sv_maj);
        }
}

svm_result svm_refinement::step(const svm_data & min_sample, const svm_data & maj_sample) {
        std::cout << "min hierarchy " << min_hierarchy->size()
                  << " -- maj hierarchy " << maj_hierarchy->size() << std::endl;

        uncoarse();

        std::cout << "min hierarchy " << min_hierarchy->size()
                  << " -- maj hierarchy " << maj_hierarchy->size() << std::endl;

        std::cout << "current level nodes"
                  << " min " << neighbors_min.size()
                  << " maj " << neighbors_maj.size()
                  << std::endl;

        svm_instance instance;
        instance.read_problem(neighbors_min, neighbors_maj);
        svm_solver solver(instance);

        if (neighbors_min.size() + neighbors_maj.size() < this->num_skip_ms) {
                if (training_inherit) {
                        result = solver.train_refinement(min_sample, maj_sample, inherit_ud, result.best().C_log, result.best().gamma_log);
                } else {
                        result = solver.train_initial(min_sample, maj_sample);
                }
        } else {
                // std::cout << "test over result range" << std::endl;
                // std::vector<svm_param> refine_range = result.all_params();
                // result = solver.train_range(refine_range, min_sample, maj_sample);

                std::cout << "skip training just use log C=" << result.best().C_log
                          << " log gamma=" << result.best().gamma_log << std::endl;
                solver.set_C(result.best().C);
                solver.set_gamma(result.best().gamma);
                solver.train();
                svm_summary s = solver.predict_validation_data(min_sample, maj_sample);
                s.print();
                std::vector<svm_summary> vec;
                vec.push_back(s);
                result = svm_result(vec, instance);
        }

        return result;
}

svm_data svm_refinement::get_SV_neighbors(const graph_access & G,
                                          const CoarseMapping & coarse_mapping,
                                          const std::vector<NodeID> & sv) {
        svm_data neighbors;
        std::unordered_set<NodeID> sv_set{sv.begin(), sv.end()};

        forall_nodes(G, node) {
                NodeID coarse_node = coarse_mapping[node];
                if (sv_set.find(coarse_node) != sv_set.end()) {
                        svm_feature feature = svm_convert::feature_to_node(G.getFeatureVec(node));
                        neighbors.push_back(std::move(feature));
                }
        } endfor

        std::cout << "uncoarsened nodes " << G.number_of_nodes()
                  << " SV " << sv.size()
                  << " resulting neighbors " << neighbors.size()
                  << std::endl;

        return neighbors;
}

