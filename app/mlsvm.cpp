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

        balance_configuration bc;
        bc.configurate_balance( partition_config, G_maj);

        partition_config.stop_rule = STOP_RULE_FIXED;
        partition_config.sep_num_vert_stop = 500;
        partition_config.matching_type = CLUSTER_COARSENING;

        if( partition_config.cluster_upperbound == std::numeric_limits< NodeWeight >::max()/2 ) {
                std::cout <<  "no size-constrained specified" << std::endl;
        } else {
                std::cout <<  "size-constrained set to " <<  partition_config.cluster_upperbound << std::endl;
        }

        partition_config.upper_bound_partition = partition_config.cluster_upperbound+1;
        partition_config.cluster_coarsening_factor = 1;

        std::cout << "config.k : "  << partition_config.k << std::endl;
        std::cout << "config.mode_node_separators : " << partition_config.mode_node_separators  << std::endl;
        std::cout << "config.stop_rule: " << partition_config.stop_rule << std::endl;
        std::cout << "config.matching_type: " << partition_config.matching_type << std::endl;
        std::cout << "config.graph_allready_partitioned: " << partition_config.graph_allready_partitioned << std::endl;

        coarsening coarsen;
        graph_hierarchy hierarchy;

        t.restart();
        coarsen.perform_coarsening(partition_config, G_maj, hierarchy);

        std::cout << "coarsening time: " << t.elapsed() << std::endl;
        std::cout << "hierarchy size: " << hierarchy.size() << std::endl;

        return 0;
}
