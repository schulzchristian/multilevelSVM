#include <iostream>
#include <unordered_set>

#include "svm_refinement.h"
#include "svm_convert.h"
#include "param_search.h"
#include "timer.h"

svm_refinement::svm_refinement() {
}

svm_refinement::~svm_refinement() {
}

svm_result svm_refinement::main(graph_hierarchy & min_hierarchy, graph_hierarchy & maj_hierarchy,
                                svm_solver & final_solver, const svm_result & initial_result,
                                const svm_data & min_sample, const svm_data & maj_sample,
                                const svm_data & min_test, const svm_data & maj_test) {
        svm_data neighbors_min = svm_convert::gaccess_to_nodes(*min_hierarchy.get_coarsest());
        svm_data neighbors_maj;
        std::vector<NodeID> sv_min;
        std::vector<NodeID> sv_maj;
        svm_result result = initial_result;
        bool training_inherit = false;

        svm_data min_new_sample = min_sample;
        svm_data maj_new_sample = maj_sample;

        timer t;

        while(!min_hierarchy.isEmpty() || !maj_hierarchy.isEmpty()) {
                sv_min = result[0].SV_min;
                sv_maj = result[0].SV_maj;


                if (!min_hierarchy.isEmpty() && min_hierarchy.size() >= maj_hierarchy.size()) {
                        std::cout << "minority uncoarsed" << std::endl;
                        graph_access* G_min = min_hierarchy.pop_finer_and_project();
                        CoarseMapping* mapping_min = min_hierarchy.get_mapping_of_current_finer();
                        neighbors_min = get_SV_neighbors(*G_min, *mapping_min, sv_min);
                        // delete G_min;
                        training_inherit = true; // after the first uncoarsening of the min data inherit params
                }
                if (!maj_hierarchy.isEmpty()) {
                        std::cout << "majority uncoarsed" << std::endl;
                        graph_access* G_maj = maj_hierarchy.pop_finer_and_project();
                        CoarseMapping* mapping_maj = maj_hierarchy.get_mapping_of_current_finer();
                        neighbors_maj = get_SV_neighbors(*G_maj, *mapping_maj, sv_maj);
                        // delete G_maj;
                }

                std::cout << "minority hierarchy " << min_hierarchy.size()
                          << " majority hierarchy " << maj_hierarchy.size() << std::endl;

                std::cout << "VVVVVVVVVVVVVV -- read problem -"
                          << " min " << neighbors_min.size()
                          << " maj " << neighbors_maj.size()
                          << " -- VVVVVVVVVVVVVV" << std::endl;
                svm_solver solver;
                solver.read_problem(neighbors_min, neighbors_maj);

                // std::cout << "test over result range" << std::endl;
                // std::vector<svm_param> refine_range = param_search::from_result(result);
                // result = solver.train_range(refine_range, min_new_sample, maj_new_sample);

                // auto params = param_search::mlsvm_method(-10, 10, -10, 10, false, true, result[0].C_log, result[0].gamma_log);
                if (neighbors_min.size() + neighbors_maj.size() < 10000) {
                        result = solver.train_initial(min_new_sample, maj_new_sample, training_inherit, result[0].C_log, result[0].gamma_log);
                } else {
                        std::cout << "skip training just use logC=" << result[0].C_log
                                  << " log gamma=" << result[0].gamma_log << std::endl;
                        solver.set_C(result[0].C);
                        solver.set_gamma(result[0].gamma);
                        solver.train();
                        result.clear();
                        svm_summary s = solver.predict_validation_data(min_new_sample, maj_new_sample);
                        result.push_back(s);
                }



                std::cout << "final validation on hole training data:" << std::endl;
                svm_summary test_summary = solver.predict_validation_data(min_test, maj_test);
                test_summary.print();


                final_solver = std::move(solver);

                // min_new_sample = svm_convert::sample_from_graph(*(min_hierarchy.get_finest()), 0.2f);
                // maj_new_sample = svm_convert::sample_from_graph(*(maj_hierarchy.get_finest()), 0.2f);

                std::cout << "AAAAAAAAAA -- done one uncoarsening step in " << t.elapsed() << "-- AAAAAAAAAA" << std::endl;
                t.restart();
        }


        svm_summary best = result[0];
        // final_solver = svm_solver();
        // final_solver.read_problem(*min_hierarchy.get_coarsest(), *maj_hierarchy.get_coarsest());
        final_solver.set_C(best.C);
        final_solver.set_gamma(best.gamma);
        final_solver.train();
        std::cout << "final validation on hole training data:" << std::endl;
        svm_summary test_summary = final_solver.predict_validation_data(min_test, maj_test);
        test_summary.print();

        return result;
}

svm_data svm_refinement::get_SV_neighbors(const graph_access & G,
                                          const CoarseMapping & coarse_mapping,
                                          const std::vector<NodeID> & sv) {

        std::cout << "G nodes " << G.number_of_nodes()
                  << " coarsemapping " << coarse_mapping.size()
                  << " SV " << sv.size()
                  << std::endl;

        svm_data neighbors;
        std::unordered_set<NodeID> sv_set{sv.begin(), sv.end()};

        forall_nodes(G, node) {
                NodeID coarse_node = coarse_mapping[node];
                if (sv_set.find(coarse_node) != sv_set.end()) {
                        svm_feature feature = svm_convert::feature_to_node(G.getFeatureVec(node));
                        neighbors.push_back(std::move(feature));
                }
        } endfor

        return neighbors;
}
