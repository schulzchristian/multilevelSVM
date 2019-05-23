#include <argtable2.h>
#include <iostream>
#include <math.h>
#include <regex.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "data_structure/graph_access.h"
#include "data_structure/graph_hierarchy.h"
#include "io/graph_io.h"
#include "partition/partition_config.h"
#include "partition/coarsening/coarsening.h"
#include "svm/svm_solver.h"
#include "svm/svm_solver_libsvm.h"
#include "svm/svm_solver_thunder.h"
#include "svm/svm_convert.h"
#include "svm/k_fold.h"
#include "svm/k_fold_build.h"
#include "svm/k_fold_import.h"
#include "svm/svm_refinement.h"
#include "svm/svm_result.h"
#include "svm/results.h"
#include "tools/timer.h"
#include "parse_parameters.h"

void print_null(const char *s) {}

void kfold_instance(PartitionConfig& partition_config, std::unique_ptr<k_fold>& kfold, results& results) {
        timer t;
        graph_access *G_min = kfold->getMinGraph();
        graph_access *G_maj = kfold->getMajGraph();

        G_min->set_partition_count(partition_config.k);
        G_maj->set_partition_count(partition_config.k);

        std::cout << "graph -"
                        << " min: " << G_min->number_of_nodes()
                        << " maj: " << G_maj->number_of_nodes() << std::endl;

        std::cout << "test -"
                        << " min: " << kfold->getMinTestData()->size()
                        << " maj: " << kfold->getMajTestData()->size() << std::endl;

        auto min_validation = svm_convert::sample_from_graph(*G_min, partition_config.sample_percent);
        auto maj_validation = svm_convert::sample_from_graph(*G_maj, partition_config.sample_percent);

        std::cout << "sample -"
                        << " min: " << min_validation.size()
                        << " maj: " << maj_validation.size() << std::endl;


        // ------------- TRAINING -----------------

        t.restart();

        svm_instance instance;
        instance.read_problem(*G_min, *G_maj);

        svm_solver_thunder solver(instance);
        auto result = solver.train_initial(min_validation, maj_validation);

        auto train_time = t.elapsed();
        std::cout << "train time: " << train_time << std::endl;

        auto summary = result.best();

        results.setFloat("AC", summary.Acc);
        results.setFloat("SN", summary.Sens);
        results.setFloat("SP", summary.Spec);
        results.setFloat("GM", summary.Gmean);
        results.setFloat("F1", summary.F1);

        t.restart();


        // ------------- TEST -----------------

        std::cout << "validation on test data:" << std::endl;
        auto summary_test = solver.build_summary(*kfold->getMinTestData(), *kfold->getMajTestData());
        auto test_time = t.elapsed();
        std::cout << "test time " << test_time << std::endl;
        results.setFloat("\tTEST_TIME", test_time);

        summary_test.print();
        results.setFloat("AC_TEST", summary_test.Acc);
        results.setFloat("SN_TEST", summary_test.Sens);
        results.setFloat("SP_TEST", summary_test.Spec);
        results.setFloat("GM_TEST", summary_test.Gmean);
        results.setFloat("F1_TEST", summary_test.F1);
}

// from https://stackoverflow.com/questions/40550730/how-to-implement-timeout-for-function-in-c
void kfold_timeout(int timeout_secs, PartitionConfig& partition_config, std::unique_ptr<k_fold>& kfold, results& results) {
        std::mutex m;
        std::condition_variable cv;

        std::thread t([&cv, &partition_config, &kfold, &results]()
                      {
                              kfold_instance(partition_config, kfold, results);
                              cv.notify_one();
                      });

        t.detach();

        {
                std::unique_lock<std::mutex> l(m);
                if(cv.wait_for(l, std::chrono::seconds(timeout_secs)) == std::cv_status::timeout) {
                        throw std::runtime_error("Timeout");
                }
        }
}

int main(int argn, char *argv[]) {
        PartitionConfig partition_config;
        std::string filename;

        bool suppress_output   = false;

        int ret_code = parse_parameters(argn, argv, partition_config, filename, suppress_output);

        if(ret_code) {
                return -1;
        }

        // disable libsvm output
        svm_set_print_string_function(&print_null);
	// disable thundersvm output
	el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, "false");

	if (partition_config.n_cores > 0) {
		omp_set_num_threads(partition_config.n_cores);
	}

	partition_config.LogDump(stdout);
        partition_config.k = 1;
        partition_config.cluster_coarsening_factor = 1;
        partition_config.stop_rule = STOP_RULE_FIXED;

        std::cout << "cores: " << partition_config.n_cores << std::endl;
        std::cout << "num_experiments: " << partition_config.num_experiments << std::endl;
        std::cout << "kfold_iterations: " << partition_config.kfold_iterations << std::endl;
        std::cout << "import_kfold: " << partition_config.import_kfold << std::endl;
        std::cout << "bidirectional: " << partition_config.bidirectional << std::endl;
        std::cout << "stop rule: " << partition_config.stop_rule << std::endl;
        std::cout << "fix stop vertices: " << partition_config.fix_num_vert_stop << std::endl;
        std::cout << "matching type: " << partition_config.matching_type << std::endl;
        std::cout << "cluster_upperbound: " << partition_config.cluster_upperbound << std::endl;
        std::cout << "upper_bound_partition: " << partition_config.upper_bound_partition << std::endl;
        std::cout << "label_iterations: " << partition_config.label_iterations << std::endl;
        std::cout << "node_ordering: " << partition_config.node_ordering << std::endl;
        std::cout << "diameter_upperbound: " << partition_config.diameter_upperbound << std::endl;
        std::cout << "num_skip_ms: " << partition_config.num_skip_ms << std::endl;
        std::cout << "inherit_ud: " << partition_config.inherit_ud << std::endl;
        std::cout << "timeout: " << partition_config.timeout << std::endl;
        std::cout << "seed: " << partition_config.seed << std::endl;

        random_functions::setSeed(partition_config.seed);

        results results;

        for (int r = 0; r < partition_config.num_experiments; r++) {
                std::cout << " \\/\\/\\/\\/\\/\\/\\/\\/\\/ EXPERIMENT " << r << " \\/\\/\\/\\/\\/\\/\\/" << std::endl;

                std::unique_ptr<k_fold> kfold;

                if(partition_config.import_kfold) {
                        kfold.reset(new k_fold_import(partition_config, r, filename));
                } else {
                        kfold.reset(new k_fold_build(partition_config, filename));
                }

                timer t_all;
                double kfold_io_time = 0;

                while (kfold->next(kfold_io_time)) {
                        results.next();
                        bool timedout = false;

                        auto kfold_time = t_all.elapsed() - kfold_io_time;
                        std::cout << "fold time: " << kfold_time << std::endl;
                        results.setFloat("KFOLD_TIME", kfold_time);

                        try {
                                if (partition_config.timeout > 0) {
                                        kfold_timeout(partition_config.timeout, partition_config, kfold, results);
                                }
                                else {
                                        kfold_instance(partition_config, kfold, results);
                                }
                        }
                        catch(std::runtime_error& e) {
                                std::cout << e.what() << std::endl;
                                timedout = true;
                        }

                        if(timedout) {
                                std::cout << "kfold timeout reached... quitting" << std::endl;
                                exit(123);
                        }

                        auto time_all = t_all.elapsed();
                        auto time_iteration = time_all - kfold_io_time;

                        std::cout << "iteration time: " << time_iteration << std::endl;
                        results.setFloat("TIME", time_iteration);

                        kfold_io_time = 0;
                        t_all.restart();
                }
        }

        results.print();

        return 0;
}
