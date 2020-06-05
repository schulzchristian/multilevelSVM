#include <argtable2.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <regex.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <memory>
#include <svm.h>

#include "partition/partition_config.h"
#include "svm/svm_solver_libsvm.h"
#include "svm/svm_solver_thunder.h"
#include "svm/svm_convert.h"
#include "svm/svm_result.h"
#include "svm/results.h"
#include "tools/timer.h"
#include "parse_parameters.h"

#include <thundersvm/util/log.h>
void print_null(const char *s) {}

#ifndef SVM_SOLVER
#define SVM_SOLVER svm_solver_thunder
#define SVM_MODEL SVC
// #define SVM_SOLVER svm_solver_libsvm
// #define SVM_MODEL svm_model
#endif

int main(int argn, char *argv[]) {
	PartitionConfig partition_config;

        if(parse_parameters(argn, argv, partition_config)) {
                return -1;
        }

        // disable libsvm output
        svm_set_print_string_function(&print_null);
	// disable thundersvm output
	el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, "false");

	partition_config.apply();
	partition_config.print();

	/*
	  We dream off a tool that takes a kasvm-model
	  (maybe only the exported libsvm model)
	  and evaluates it automaticaly on some data
	  therefore the tool has to be aware of the transformations
	  done by the 'prepare' tool and also apply these to the data.
	*/	
}
