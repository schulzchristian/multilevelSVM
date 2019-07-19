#include <argtable2.h>
#include <iostream>
#include <iomanip>
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
#include "svm/k_fold_once.h"
#include "svm/ud_refinement.h"
#include "svm/svm_result.h"
#include "svm/results.h"
#include "tools/timer.h"
#include "parse_parameters.h"

void print_null(const char *s) {}

#define SVM_SOLVER svm_solver_thunder
#define SVM_MODEL SVC

void kfold_instance(PartitionConfig& partition_config, std::unique_ptr<k_fold>& kfold, results& results) {
        timer t;
        graph_access *G_min = kfold->getMinGraph();
        graph_access *G_maj = kfold->getMajGraph();

        G_min->set_partition_count(partition_config.k);
        G_maj->set_partition_count(partition_config.k);

        std::cout << "graph -"
                        << " min: " << G_min->number_of_nodes()
                        << " maj: " << G_maj->number_of_nodes() << std::endl;

        std::cout << "val -"
                        << " min: " << kfold->getMinValData()->size()
                        << " maj: " << kfold->getMajValData()->size() << std::endl;

        std::cout << "test -"
                        << " min: " << kfold->getMinTestData()->size()
                        << " maj: " << kfold->getMajTestData()->size() << std::endl;


        // ------------- TRAINING -----------------

        t.restart();

        svm_instance instance;
        instance.read_problem(*G_min, *G_maj);

        SVM_SOLVER solver(instance);
        auto result = ud_refinement<SVM_MODEL>::train_ud(solver, *kfold->getMinValData(), *kfold->getMajValData());

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
	partition_config.validation_seperate = true;

        if(parse_parameters(argn, argv, partition_config)) {
                return -1;
        }

        // disable libsvm output
        svm_set_print_string_function(&print_null);
	// disable thundersvm output
	el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, "false");

        partition_config.k = 1;
        partition_config.cluster_coarsening_factor = 1;
        partition_config.stop_rule = STOP_RULE_FIXED;

	partition_config.apply();
	partition_config.print();

        results results;

        for (int exp = 0; exp < partition_config.num_experiments; exp++) {
                std::cout << " \\/\\/\\/\\/\\/\\/\\/\\/\\/ EXPERIMENT " << exp << " \\/\\/\\/\\/\\/\\/\\/" << std::endl;

                std::unique_ptr<k_fold> kfold;

                switch (partition_config.validation_type) {
		case KFOLD:
                        kfold.reset(new k_fold_build(partition_config,
						     partition_config.filename));
			break;
		case KFOLD_IMPORT:
                        kfold.reset(new k_fold_import(partition_config, exp,
						      partition_config.filename));
			break;
		case ONCE:
			kfold.reset(new k_fold_once(partition_config,
						    partition_config.filename));
			break;
		// case TRAIN_TEST_SPLIT:
			// kfold.reset(new k_fold_traintest(partition_config,
			// 				 partition_config.filename,
			// 				 partition_config.testname));
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
                                        kfold_timeout(partition_config.timeout,
						      partition_config, kfold, results);
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

			std::cout << "iteration time: " << std::setprecision(4) << std::fixed
				  << time_iteration << std::endl;
                        results.setFloat("TIME", time_iteration);

                        kfold_io_time = 0;
                        t_all.restart();
                }
        }

        results.print();

        return 0;
}
