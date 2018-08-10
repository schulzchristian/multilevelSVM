#include <argtable2.h>
#include <iostream>
#include <math.h>
#include <regex.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <memory>

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
#include "svm/k_fold_build.h"
#include "svm/k_fold_import.h"
#include "svm/svm_refinement.h"
#include "svm/svm_result.h"
#include "svm/results.h"

void print_null(const char *s) {}

int main(int argn, char *argv[]) {

        PartitionConfig partition_config;
        std::string filename;

        bool is_graph_weighted = false;
        bool suppress_output   = false;
        bool recursive         = false;

        int ret_code = parse_parameters(argn, argv, partition_config, filename, is_graph_weighted, suppress_output, recursive);

        if(ret_code) {
                return -1;
        }

        // disable libsvm output
        svm_set_print_string_function(&print_null);

        partition_config.LogDump(stdout);
        partition_config.k = 1;
        // partition_config.cluster_upperbound = std::numeric_limits<NodeID>::max()/2;
        partition_config.cluster_coarsening_factor = 1;
        partition_config.upper_bound_partition = std::numeric_limits<NodeID>::max()/2;
        partition_config.stop_rule = STOP_RULE_FIXED;
        partition_config.matching_type = CLUSTER_COARSENING;
        partition_config.sep_num_vert_stop = partition_config.fix_num_vert_stop;
        std::cout << "seed: " << partition_config.seed << std::endl;
        // std::cout << "using grid_search: " << partition_config.seed << std::endl;
        std::cout << "timeout: " << partition_config.timeout << std::endl;


        random_functions::setSeed(partition_config.seed);

        results results;

        for (int r = 0; r < partition_config.num_experiments; r++) {
        std::cout << " \\/\\/\\/\\/\\/\\/\\/\\/\\/ EXPERIMENT " << r << " \\/\\/\\/\\/\\/\\/\\/" << std::endl;

        std::unique_ptr<k_fold> kfold;

        if(partition_config.import_kfold) {
                kfold.reset(new k_fold_import(r, partition_config.kfold_iterations, filename));
        } else {
                kfold.reset(new k_fold_build(partition_config.num_nn, partition_config.kfold_iterations, filename));
        }

        timer t_all;
        timer t;
        double kfold_io_time = 0;

        while (kfold->next(kfold_io_time)) {
        results.next();
        graph_access *G_min = kfold->getMinGraph();
        graph_access *G_maj = kfold->getMajGraph();

        auto kfold_time = t.elapsed() - kfold_io_time;
        std::cout << "fold time: " << kfold_time << std::endl;
        results.setFloat("KFOLD_TIME", kfold_time);
        kfold_io_time = 0;

        G_min->set_partition_count(partition_config.k);
        G_maj->set_partition_count(partition_config.k);

        std::cout << "graph -"
                        << " min: " << G_min->number_of_nodes()
                        << " maj: " << G_maj->number_of_nodes() << std::endl;

        std::cout << "test -"
                        << " min: " << kfold->getMinTestData()->size()
                        << " maj: " << kfold->getMajTestData()->size() << std::endl;

        auto min_sample = svm_convert::sample_from_graph(*(kfold->getMinGraph()), partition_config.sample_percent);
        auto maj_sample = svm_convert::sample_from_graph(*(kfold->getMajGraph()), partition_config.sample_percent);

        std::cout << "sample -"
                        << " min: " << min_sample.size()
                        << " maj: " << maj_sample.size() << std::endl;


        // ------------- TRAINING -----------------

        t.restart();

        svm_instance instance;
        instance.read_problem(*(kfold->getMinGraph()), *(kfold->getMajGraph()));

        svm_solver solver(instance);
        svm_result result = solver.train_grid(min_sample, maj_sample);

        auto train_time = t.elapsed();
        std::cout << "train time: " << train_time << std::endl;

        svm_summary summary = result.best();

        results.setFloat("AC", summary.Acc);
        results.setFloat("SN", summary.Sens);
        results.setFloat("SP", summary.Spec);
        results.setFloat("GM", summary.Gmean);
        results.setFloat("F1", summary.F1);

        t.restart();

        std::cout << "validation on test data:" << std::endl;
        svm_summary summary_test = solver.build_summary(*kfold->getMinTestData(), *kfold->getMajTestData());
        auto test_time = t.elapsed();
        std::cout << "test time " << test_time << std::endl;
        results.setFloat("\tTEST_TIME", test_time);

        summary_test.print();
        results.setFloat("AC_TEST", summary_test.Acc);
        results.setFloat("SN_TEST", summary_test.Sens);
        results.setFloat("SP_TEST", summary_test.Spec);
        results.setFloat("GM_TEST", summary_test.Gmean);
        results.setFloat("F1_TEST", summary_test.F1);

        // ------------- END --------------
        auto time_all = t_all.elapsed();

        auto time_iteration = time_all - kfold_io_time;

        std::cout << "iteration time: " << time_iteration << std::endl;

        results.setFloat("TIME", time_iteration);

        if(partition_config.timeout != 0 && time_iteration > partition_config.timeout) {
                std::cout << "timeout reached exiting..." << std::endl;
                exit(123);
        }

        t_all.restart();
        t.restart();
        }
        }

        results.print();

        return 0;
}
