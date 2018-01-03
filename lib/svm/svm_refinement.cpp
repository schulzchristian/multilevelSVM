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
                                const svm_data & min_sample, const svm_data & maj_sample) {
        graph_access* G_min;
        graph_access* G_maj;
        CoarseMapping* mapping_min;
        CoarseMapping* mapping_maj;
        svm_data neighbors_min;
        svm_data neighbors_maj;
        std::vector<NodeID> sv_min;
        std::vector<NodeID> sv_maj;
        svm_result result = initial_result;
        graph_access* min_delete = nullptr;
        graph_access* maj_delete = nullptr;

        timer t;

        while(!min_hierarchy.isEmpty() || !maj_hierarchy.isEmpty()) {
                sv_min = result[0].SV_min;
                sv_maj = result[0].SV_maj;

                if (!min_hierarchy.isEmpty()) {
                        G_min = min_hierarchy.pop_finer_and_project();
                        mapping_min = min_hierarchy.get_mapping_of_current_finer();
                        neighbors_min = get_SV_neighbors(*G_min, *mapping_min, sv_min);
                }
                if (!maj_hierarchy.isEmpty()) {
                        G_maj = maj_hierarchy.pop_finer_and_project();
                        mapping_maj = maj_hierarchy.get_mapping_of_current_finer();
                        neighbors_maj = get_SV_neighbors(*G_maj, *mapping_maj, sv_maj);
                }

                std::cout << "read problem -"
                          << " min " << neighbors_min.size()
                          << " maj " << neighbors_maj.size()
                          << std::endl;
                svm_solver solver;
                solver.read_problem(neighbors_min, neighbors_maj);


                // auto min_new_sample = svm_convert::sample_from_graph(*(min_hierarchy.get_finest()), 0.1f);
                // auto maj_new_sample = svm_convert::sample_from_graph(*(maj_hierarchy.get_finest()), 0.1f);

                std::cout << "test over result range" << std::endl;
                // std::vector<svm_param> refine_range = param_search::from_result(result);
                // result = solver.train_range(refine_range, min_new_sample, maj_new_sample);

                auto params = param_search::mlsvm_method(-10, 10, -10, 10, false, true, result[0].C_log, result[0].gamma_log);
                result = solver.train_range(params, min_sample, maj_sample);

                if (min_delete != nullptr) {
                        delete min_delete;
                }
                if (maj_delete != nullptr) {
                        delete maj_delete;
                }

                // final_solver = std::move(solver);
                final_solver = solver;
                solver.dont_delete();

                std::cout << "done one uncoarsening step in " << t.elapsed() << std::endl;
                t.restart();
        }


        std::cout << "best of uncoarsening" << std::endl;
        svm_summary best = result[0];
        best.print();
        final_solver.set_C(best.C);
        final_solver.set_gamma(best.gamma);
        final_solver.train();

        // delete min_coarsest;
        // delete maj_coarsest;

        return result;
}

svm_data svm_refinement::get_SV_neighbors(const graph_access & G,
                                          const CoarseMapping & coarse_mapping,
                                          const std::vector<NodeID> & sv) {

        std::cout << "G nodes " << G.number_of_nodes()
                  << " coarsemapping " << coarse_mapping.size()
                  << " SV " << sv.size()
                  << std::endl;

        // std::cout << "SVs ";
        // for (auto sv : sv) {
        //         std::cout << sv << ", ";
        // }
        // std::cout << "\n";

        svm_data neighbors;
        std::unordered_set<NodeID> sv_set{sv.begin(), sv.end()};

        std::cout << "neighbors ";
        forall_nodes(G, node) {
                NodeID coarse_node = coarse_mapping[node];
                if (sv_set.find(coarse_node) != sv_set.end()) {
                        svm_feature feature = svm_convert::feature_to_node(G.getFeatureVec(node));
                        neighbors.push_back(feature);
                        // std::cout << node << ", ";
                }
        } endfor
        // std::cout << "\n";

        return neighbors;
}
