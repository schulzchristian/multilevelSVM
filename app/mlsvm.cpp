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
#include "svm/svm_convert.h"

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

        // if( partition_config.cluster_upperbound == std::numeric_limits< NodeWeight >::max()/2 ) {
        //         std::cout <<  "no size-constrained specified" << std::endl;
        // } else {
        //         std::cout <<  "size-constrained set to " <<  partition_config.cluster_upperbound << std::endl;
        // }

        partition_config.upper_bound_partition = partition_config.cluster_upperbound+1;
        partition_config.cluster_coarsening_factor = 1;

        // -------- end

        partition_config.stop_rule = STOP_RULE_FIXED;
        partition_config.matching_type = CLUSTER_COARSENING;
        partition_config.sep_num_vert_stop = partition_config.fix_num_vert_stop;

        std::cout << "fix stop vertices: " << partition_config.fix_num_vert_stop << std::endl;

        coarsening coarsen;
        graph_hierarchy maj_hierarchy;
        graph_hierarchy min_hierarchy;

        t.restart();

        coarsen.perform_coarsening(partition_config, G_maj, maj_hierarchy);
        coarsen.perform_coarsening(partition_config, G_min, min_hierarchy);

        std::cout << "coarsening time: " << t.elapsed() << std::endl;
        std::cout << "coarse min nodes: " << min_hierarchy.get_coarsest()->number_of_nodes() << std::endl;
        std::cout << "coarse maj nodes: " << maj_hierarchy.get_coarsest()->number_of_nodes() << std::endl;

        t.restart();

        std::vector<std::vector<svm_node>> maj_sample = svm_convert::convert_sample_to_nodes(G_maj, 0.1);
        std::vector<std::vector<svm_node>> min_sample = svm_convert::convert_sample_to_nodes(G_min, 0.1);

        std::cout << "maj_sample: " << maj_sample.size() << std::endl;
        std::cout << "min_sample: " << min_sample.size() << std::endl;

        svm_solver solver;
        solver.read_problem(*maj_hierarchy.get_coarsest(), *min_hierarchy.get_coarsest());
        solver.train_initial(maj_sample, min_sample);

        std::cout << "initial training time: " << t.elapsed() << std::endl;


        std::cout << "validation on hole training data:" << std::endl;
        svm_summary summary = solver.predict_validation_data(svm_convert::gaccess_to_nodes(G_maj), svm_convert::gaccess_to_nodes(G_min));

        summary.print();

        return 0;
}
