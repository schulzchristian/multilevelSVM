#include <argtable2.h>
#include <iostream>
#include <math.h>
#include <regex.h>
#include <sstream>
#include <stdio.h>
#include <string.h> 

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
    return 0;
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

  partition_config.stop_rule = STOP_RULE_SIMPLE;
  partition_config.sep_num_vert_stop = 500;

  partition_config.k = 10;

  coarsening coarsen;

  graph_hierarchy hierarchy;

  coarsen.perform_coarsening(partition_config, G_maj, hierarchy);

  std::cout << "hierarchy size: " << hierarchy.size() << std::endl;


  return 0;
}
