#include <argtable2.h>
#include <iostream>
#include <math.h>
#include <regex.h>
#include <sstream>
#include <stdio.h>
#include <string.h>

#include "balance_configuration.h"
#include "data_structure/graph_access.h"
#include "graph_io.h"
#include "parse_parameters.h"
#include "partition/partition_config.h"
#include "coarsening/coarsening.h"
#include "data_structure/graph_hierarchy.h"
#include "timer.h"
#include "svm/svm_solver.h"
#include "svm/svm_convert.h"
#include "svm/k_fold.h"
#include "svm/svm_refinement.h"
#include "svm/svm_result.h"

void print_null(const char *s) {}

int main(int argn, char *argv[]) {

        PartitionConfig partition_config;
        std::string filename;

        bool is_graph_weighted = false;
        bool suppress_output   = false;
        bool recursive         = false;

        int ret_code = parse_parameters(argn, argv,
                                        partition_config,
                                        filename,
                                        is_graph_weighted,
                                        suppress_output,
                                        recursive);

        if(ret_code) {
                return -1;
        }


        // turn 
        svm_set_print_string_function(&print_null);

        partition_config.LogDump(stdout);

        partition_config.k = 1;

        // -------- copied from label_propagation

        // if( partition_config.cluster_upperbound == std::numeric_limits< NodeWeight >::max()/2 ) {
        //         std::cout <<  "no size-constrained specified" << std::endl;
        // } else {
        //         std::cout <<  "size-constrained set to " <<  partition_config.cluster_upperbound << std::endl;
        // }


        // -------- end

        partition_config.cluster_upperbound = 100;
        partition_config.cluster_coarsening_factor = 1;
        partition_config.upper_bound_partition = std::numeric_limits<NodeID>::max()/2;
        partition_config.stop_rule = STOP_RULE_FIXED;
        partition_config.matching_type = CLUSTER_COARSENING;
        partition_config.sep_num_vert_stop = partition_config.fix_num_vert_stop;
        std::cout << "fix stop vertices: " << partition_config.fix_num_vert_stop << std::endl;
        std::cout << "label_iterations: " << partition_config.label_iterations << std::endl;


        k_fold kfold(5, filename);

        timer t_all;
        timer t;

        while (kfold.next()) {
                graph_access *G_min = kfold.getMinGraph();
                graph_access *G_maj = kfold.getMajGraph();

                auto kfold_time = t.elapsed();

                std::cout << "fold time: " << kfold_time << std::endl;
                kfold.setResult("KFOLD_TIME", kfold_time);

                G_min->set_partition_count(partition_config.k);
                G_maj->set_partition_count(partition_config.k);

                std::cout << "graph -"
                          << " min: " << G_min->number_of_nodes()
                          << " maj: " << G_maj->number_of_nodes() << std::endl;

                std::cout << "test -"
                          << " min: " << kfold.getMinTestData()->size()
                          << " maj: " << kfold.getMajTestData()->size() << std::endl;

                auto min_sample = svm_convert::sample_from_graph(*(kfold.getMinGraph()), 0.1f);
                auto maj_sample = svm_convert::sample_from_graph(*(kfold.getMajGraph()), 0.1f);

                std::cout << "sample -"
                          << " min: " << min_sample.size()
                          << " maj: " << maj_sample.size() << std::endl;


                // ------------- COARSENING -----------------

                t.restart();

                coarsening coarsen;
                graph_hierarchy min_hierarchy;
                graph_hierarchy maj_hierarchy;

                // balance_configuration::configurate_balance(partition_config, *G_min);
                coarsen.perform_coarsening(partition_config, *G_min, min_hierarchy);

                // balance_configuration::configurate_balance(partition_config, *G_maj);
                coarsen.perform_coarsening(partition_config, *G_maj, maj_hierarchy);

                auto coarsening_time = t.elapsed();
                std::cout << "coarsening time: " << coarsening_time << std::endl
                          << "coarse nodes - min: " << min_hierarchy.get_coarsest()->number_of_nodes()
                          << " maj: " << maj_hierarchy.get_coarsest()->number_of_nodes() << std::endl;

                kfold.setResult("COARSE_TIME", coarsening_time);
                kfold.setResult("COARSE_MIN", min_hierarchy.get_coarsest()->number_of_nodes());
                kfold.setResult("COARSE_MAJ", maj_hierarchy.get_coarsest()->number_of_nodes());
                kfold.setResult("HIERARCHY_MIN_SIZE", min_hierarchy.size());
                kfold.setResult("HIERARCHY_MAJ_SIZE", maj_hierarchy.size());

                // ------------- INITIAL TRAINING -----------------

                t.restart();

                svm_instance initial_instance;
                initial_instance.read_problem(*min_hierarchy.get_coarsest(), *maj_hierarchy.get_coarsest());

                svm_solver init_solver(initial_instance);
                svm_result initial_result = init_solver.train_initial(min_sample, maj_sample);

                auto init_train_time = t.elapsed();
                std::cout << "init train time: " << init_train_time << std::endl;

                svm_summary initial_summary = initial_result.best();
                kfold.setResult("\tINIT_TRAIN_TIME", init_train_time);
                kfold.setResult("INIT_AC  ", initial_summary.Acc);
                kfold.setResult("INIT_GM  ", initial_summary.Gmean);

                std::cout << "inital validation on hole training data:" << std::endl;
                svm_summary initial_test_summary = init_solver.predict_validation_data(*kfold.getMinTestData(), *kfold.getMajTestData());
                initial_test_summary.print();
                kfold.setResult("INIT_AC_TRAIN", initial_test_summary.Acc);
                kfold.setResult("INIT_GM_TRAIN", initial_test_summary.Gmean);

                // ------------- REFINEMENT -----------------

                t.restart();

                svm_refinement refinement(min_hierarchy, maj_hierarchy, initial_result);

                // std::vector<std::pair<svm_summary, svm_solver>> best_results;

                while (!refinement.is_done()) {
                  timer t_ref;

                  // min_sample = svm_convert::sample_from_graph(*(min_hierarchy.get_finest()), 0.2f);
                  // maj_sample = svm_convert::sample_from_graph(*(maj_hierarchy.get_finest()), 0.2f);

                  svm_result current_result = refinement.step(min_sample, maj_sample);

                  std::cout << "refinement at level " << refinement.get_level()
                            << " took " << t_ref.elapsed() << std::endl;

                  // best_results.push_back(std::make_pair(current_result.best(), current_result.instance));

                  std::ostringstream fmt_ac, fmt_gm;
                  fmt_ac << "LEVEL" << refinement.get_level() << "_AC";
                  fmt_gm << "LEVEL" << refinement.get_level() << "_GM";
                  kfold.setResult(fmt_ac.str(), current_result.best().Acc);
                  kfold.setResult(fmt_gm.str(), current_result.best().Gmean);

                  std::cout << "level " << refinement.get_level()
                            << " validation on hole training data:" << std::endl;

                  svm_solver solver(current_result.instance);
                  solver.set_C(current_result.best().C);
                  solver.set_gamma(current_result.best().gamma);
                  solver.train();
                  svm_summary final_test_summary = solver.predict_validation_data(*kfold.getMinTestData(), *kfold.getMajTestData());
                  final_test_summary.print();

                  fmt_ac << "_TEST";
                  fmt_gm << "_TEST";
                  kfold.setResult(fmt_ac.str(), final_test_summary.Acc);
                  kfold.setResult(fmt_gm.str(), final_test_summary.Gmean);

                }

                auto refinement_time = t.elapsed();
                std::cout << "refinement time " << refinement_time << std::endl;
                kfold.setResult("\tREFINEMENT_TIME", refinement_time);

                /*
                svm_summary best_summary = svm_solver::select_best_model(best_results);
                kfold.setResult("BEST_AC", best_summary.Acc);
                kfold.setResult("BEST_GM", best_summary.Gmean);

                std::cout << "best validation on hole training data:" << std::endl;
                svm_summary best_summary_test = solver.predict_validation_data(*kfold.getMinTestData(), *kfold.getMajTestData());
                best_summary_test.print();

                kfold.setResult("BEST_AC_TEST", best_summary_test.Acc);
                kfold.setResult("BEST_GM_TEST", best_summary_test.Gmean);
                */

                // ------------- END --------------
                auto time_iteration = t_all.elapsed();

                std::cout << "iteration time: " << time_iteration << std::endl;

                kfold.setResult("TIME", time_iteration);

                t.restart();
                t_all.restart();
        }

        kfold.printAverages();


        return 0;
}
