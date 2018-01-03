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



        partition_config.LogDump(stdout);

        partition_config.k = 1;

        // -------- copied from label_propagation

        // if( partition_config.cluster_upperbound == std::numeric_limits< NodeWeight >::max()/2 ) {
        //         std::cout <<  "no size-constrained specified" << std::endl;
        // } else {
        //         std::cout <<  "size-constrained set to " <<  partition_config.cluster_upperbound << std::endl;
        // }


        // -------- end

        partition_config.stop_rule = STOP_RULE_FIXED;
        partition_config.matching_type = CLUSTER_COARSENING;
        partition_config.sep_num_vert_stop = partition_config.fix_num_vert_stop;
        std::cout << "fix stop vertices: " << partition_config.fix_num_vert_stop << std::endl;


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


                // ------------- COARSENING -----------------

                t.restart();

                coarsening coarsen;
                graph_hierarchy min_hierarchy;
                graph_hierarchy maj_hierarchy;

                balance_configuration::configurate_balance(partition_config, *G_min);
                partition_config.upper_bound_partition = partition_config.cluster_upperbound+1;
                partition_config.cluster_coarsening_factor = 1;

                coarsen.perform_coarsening(partition_config, *G_min, min_hierarchy);

                balance_configuration::configurate_balance(partition_config, *G_maj);
                partition_config.upper_bound_partition = partition_config.cluster_upperbound+1;
                partition_config.cluster_coarsening_factor = 1;

                coarsen.perform_coarsening(partition_config, *G_maj, maj_hierarchy);

                auto coarsening_time = t.elapsed();
                std::cout << "coarsening time: " << coarsening_time << std::endl
                          << "coarse nodes - min: " << min_hierarchy.get_coarsest()->number_of_nodes()
                          << " maj: " << maj_hierarchy.get_coarsest()->number_of_nodes() << std::endl;

                kfold.setResult("COARSE_TIME", coarsening_time);

                // ------------- INITIAL TRAINING -----------------

                t.restart();

                auto min_sample = svm_convert::sample_from_graph(*(kfold.getMinGraph()), 0.1f);
                auto maj_sample = svm_convert::sample_from_graph(*(kfold.getMajGraph()), 0.1f);

                std::cout << "sample -"
                          << " min: " << min_sample.size()
                          << " maj: " << maj_sample.size() << std::endl;

                svm_solver solver;
                solver.read_problem(*min_hierarchy.get_coarsest(), *maj_hierarchy.get_coarsest());
                svm_result initial_result = solver.train_initial(min_sample, maj_sample);

                auto init_train_time = t.elapsed();
                std::cout << "init train time: " << init_train_time << std::endl;
                kfold.setResult("INIT_TRAIN_TIME", init_train_time);


                std::cout << "inital validation on hole training data:" << std::endl;
                svm_summary initial_summary = solver.predict_validation_data(*kfold.getMinTestData(), *kfold.getMajTestData());

                std::cout << "init train result: ";
                initial_summary.print();

                kfold.setResult("INIT_ACC", initial_summary.Acc);
                kfold.setResult("INIT_GMEAN", initial_summary.Gmean);

                // ------------- REFINEMENT -----------------

                t.restart();

                svm_refinement refinement;
                svm_result final_result = refinement.main(min_hierarchy, maj_hierarchy,
                                                          solver, initial_result,
                                                          min_sample, maj_sample);

                auto refinement_time = t.elapsed();
                std::cout << "refinement timed " << refinement_time << std::endl;
                kfold.setResult("REFINE_TIME", refinement_time);

                std::cout << "final validation on hole training data:" << std::endl;
                svm_summary final_summary = solver.predict_validation_data(*kfold.getMinTestData(), *kfold.getMajTestData());
                final_summary.print();

                kfold.setResult("RESULT_ACC", final_summary.Acc);
                kfold.setResult("RESULT_GMEAN", final_summary.Gmean);

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
