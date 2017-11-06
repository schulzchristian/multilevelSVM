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
#include "timer.h"
#include "svm/svm_solver.h"

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
                                        suppress_output, recursive);

        if(ret_code) {
                return -1;
        }

        partition_config.LogDump(stdout);

        graph_access G_min;
        graph_access G_maj;

        timer t;
        graph_io::readGraphWeighted(G_min, filename + "_min_graph");
        graph_io::readFeatures(G_min, filename + "_min_data");
        graph_io::readGraphWeighted(G_maj, filename + "_maj_graph");
        graph_io::readFeatures(G_maj, filename + "_maj_data");
        std::cout << "io time: " << t.elapsed() << std::endl;

        std::cout << "min nodes: " << G_min.number_of_nodes() << std::endl;
        std::cout << "maj nodes: " << G_maj.number_of_nodes() << std::endl;
        std::cout << "features: " << G_min.getFeatureVec(0).size() << std::endl;

        partition_config.k = 1;
        G_maj.set_partition_count(partition_config.k);

        // -------- copied from label_propagation

        balance_configuration bc;
        bc.configurate_balance(partition_config, G_maj);

        if( partition_config.cluster_upperbound == std::numeric_limits< NodeWeight >::max()/2 ) {
                std::cout <<  "no size-constrained specified" << std::endl;
        } else {
                std::cout <<  "size-constrained set to " <<  partition_config.cluster_upperbound << std::endl;
        }

        partition_config.upper_bound_partition = partition_config.cluster_upperbound+1;
        partition_config.cluster_coarsening_factor = 1;

        // -------- end

        partition_config.stop_rule = STOP_RULE_FIXED;
        partition_config.matching_type = CLUSTER_COARSENING;

        coarsening coarsen;
        graph_hierarchy maj_hierarchy;
        graph_hierarchy min_hierarchy;

        t.restart();

        coarsen.perform_coarsening(partition_config, G_maj, maj_hierarchy);
        coarsen.perform_coarsening(partition_config, G_min, min_hierarchy);

        std::cout << "coarsening time: " << t.elapsed() << std::endl;

        t.restart();

        svm_solver solver;
        solver.read_problem(*maj_hierarchy.get_coarsest(), *min_hierarchy.get_coarsest());
        solver.train();

        std::cout << "initial training time: " << t.elapsed() << std::endl;

        int errors = 0;

        graph_access * G = maj_hierarchy.get_coarsest();
        forall_nodes((*G), n) {
                int val = solver.predict(G->getFeatureVec(n));
                if (val != -1) {
                        errors++;
                }
        } endfor

        G = min_hierarchy.get_coarsest();
        forall_nodes((*G), n) {
                int val = solver.predict(G->getFeatureVec(n));
                if (val != 1) {
                        errors++;
                }
        } endfor

        std::cout << "errors: " << errors << " of " << maj_hierarchy.get_coarsest()->number_of_nodes() + G->number_of_nodes() << std::endl;

        return 0;
}
