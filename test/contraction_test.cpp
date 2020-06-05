#include "catch.hpp"
#include "data_structure/graph_access.h"
#include "data_structure/graph_hierarchy.h"
#include "partition/coarsening/coarsening.h"
#include "partition/partition_config.h"
#include "tools/random_functions.h"
#include "io/graph_io.h"
#include "svm/svm_flann.h"
#include "svm/svm_refinement.h"
#include "svm/ud_refinement.h"
#include "svm/fix_refinement.h"
#include "svm/svm_result.h"
#include "svm/svm_solver_thunder.h"
#include "svm/svm_solver_libsvm.h"
#include "svm/svm_convert.h"

#include <iostream>
#include <iomanip>
#include <unordered_set>

std::vector<FeatureVec> gen_data(int clusters, int cluster_size, int levels, int features, float sparsity) {
	std::vector<std::vector<FeatureVec>> data(levels);

	for (int i = 0; i < clusters; i++) {
		FeatureVec feat(features);
		for (int f = 0; f < features; f++) {
			feat[f] = random_functions::nextDouble(-10, 10);
		}
		data[0].push_back(feat);
	}

	double cur_stddev = 1;
	// int num_points = cluster_size;
	for (int l = 0; l < levels - 1; l++) {
		cur_stddev *= sparsity;
		// num_points /= (l+1);
		// std::cout << "num_points: " << num_points << std::endl;
		for (FeatureVec datapoint: data[l]) {
			for (int i = 0; i < cluster_size / (l+1); i++) {
				FeatureVec feat(features);
				for (int f = 0; f < features; f++) {
					feat[f] = random_functions::nextFromNorm(datapoint[f], cur_stddev);
					// std::cout << feat[f] << ",";
				}
				// std::cout << std::endl;
				data[l+1].push_back(feat);
			}
			// std::cout << std::endl;
		}
	}

	return data.back();
}

graph_access to_graph(const std::vector<FeatureVec> & features) {
	int num_nn = 10;

        std::vector<std::vector<Edge>> edges;

        svm_flann::run_flann(features, edges, num_nn);

	graph_access graph;

	graph_io::readGraphFromVec(graph, edges, features.size() * num_nn * 2);
        graph_io::readFeatures(graph, features);

	return graph;
}

void mark_sv(graph_access & G,
	     std::vector<NodeID> & data_mapping,
	     const std::vector<NodeID> & sv) {
	std::cout << "sv size" << sv.size() << "data_mapping size: " << data_mapping.size() << std::endl;

	std::unordered_set<NodeID> sv_set;
	sv_set.reserve(sv.size());
	for (NodeID id : sv) {
		sv_set.insert(data_mapping[id]);
	}

        forall_nodes(G, node) {
                if (sv_set.find(node) != sv_set.end()) {
			G.setSVStatus(node, 2);
                }
        } endfor
}

TEST_CASE( "Generate a" ) {
	auto features_min = gen_data(3, 6, 4, 2, 0.85);
	auto features_maj = gen_data(9, 50, 2, 2, 0.85);

	auto graph_min = to_graph(features_min);
	auto graph_maj = to_graph(features_maj);

        graph_hierarchy min_hierarchy;
        graph_hierarchy maj_hierarchy;

	PartitionConfig partition_config;

        coarsening coarsen;

	// partition_config.matching_type = LOW_DIAMETER;
	// partition_config.beta = 0.4;
	partition_config.label_iterations = 10;
	// partition_config.cluster_upperbound = 20;
	partition_config.fix_num_vert_stop = 30;
        coarsen.perform_coarsening(partition_config, graph_min, min_hierarchy);
	partition_config.fix_num_vert_stop = 30;
        coarsen.perform_coarsening(partition_config, graph_maj, maj_hierarchy);

	// init train
	// disable thundersvm output
	el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, "false");
        svm_instance initial_instance;
        initial_instance.read_problem(*min_hierarchy.get_coarsest(), *maj_hierarchy.get_coarsest());
        svm_solver_thunder init_solver(initial_instance);
	auto val_min = svm_convert::graph_to_nodes(graph_min);
	auto val_maj = svm_convert::graph_to_nodes(graph_maj);
	svm_result<SVC> initial_result = ud_refinement<SVC>::train_ud(init_solver, val_min, val_maj);
	// svm_result<SVC> initial_result = fix_refinement<SVC>::train_fix(init_solver, val_min, val_maj, 7.5, -10);
	svm_summary<SVC> initial_summary = initial_result.best();

	// init mark SV
	{
	std::vector<NodeID> data_mapping_min; // default mapping
        forall_nodes((*min_hierarchy.get_coarsest()), node) {
		data_mapping_min.push_back(node);
	endfor }
	std::vector<NodeID> data_mapping_maj;
	forall_nodes((*maj_hierarchy.get_coarsest()), node) {
		data_mapping_maj.push_back(node);
	endfor }
	svm_refinement<SVC>::uncoarse_SV(*min_hierarchy.get_coarsest(), data_mapping_min, initial_summary.SV_min, data_mapping_min);
	svm_refinement<SVC>::uncoarse_SV(*maj_hierarchy.get_coarsest(), data_mapping_maj, initial_summary.SV_maj, data_mapping_maj);
	}

	// init export
	std::cout << "levels - min: " << min_hierarchy.size() << " maj: " << maj_hierarchy.size() << std::endl;

        int init_level = std::max(min_hierarchy.size(), maj_hierarchy.size());
	std::ostringstream out_graph_init;
	out_graph_init << "graph_testcase_" << init_level << ".gdf";
	std::cout << "exporting " << out_graph_init.str() << std::endl;
	graph_io::writeGraphGDF(*min_hierarchy.get_coarsest(), *maj_hierarchy.get_coarsest(), out_graph_init.str());

	// refinement
	std::cout << "begin refinement" << std::endl;
	partition_config.fix_C = initial_summary.C_log;
	partition_config.fix_gamma = initial_summary.gamma_log;
	fix_refinement<SVC> refinement = fix_refinement<SVC>(min_hierarchy, maj_hierarchy, initial_result, partition_config);

        while (!refinement.is_done()) {
                auto current_result = refinement.step(val_min, val_maj);
		auto current_summary = current_result.best();

		mark_sv(*refinement.G_min, refinement.data_mapping_min, current_summary.SV_min);
		mark_sv(*refinement.G_maj, refinement.data_mapping_maj, current_summary.SV_maj);

		std::cout << "levels - min: " << min_hierarchy.size() << " maj: " << maj_hierarchy.size() << std::endl;
		std::ostringstream out_graph;
		out_graph << "graph_testcase_" << refinement.get_level() << ".gdf";
		std::cout << "exporting " << out_graph.str() << std::endl;
		graph_io::writeGraphGDF(*refinement.G_min, *refinement.G_maj, out_graph.str());
	}



	std::cout << "full train" << std::endl;

        svm_instance full_instance;
        full_instance.read_problem(graph_min, graph_maj);
        svm_solver_thunder full_solver(full_instance);
	svm_result<SVC> full_result = fix_refinement<SVC>::train_fix(full_solver, val_min, val_maj, initial_summary.C_log, initial_summary.gamma_log);
	svm_summary<SVC> full_summary = full_result.best();


	{
	std::vector<NodeID> data_mapping_min; // default mapping
        forall_nodes((graph_min), node) {
		data_mapping_min.push_back(node);
	endfor }
	std::vector<NodeID> data_mapping_maj;
	forall_nodes((graph_maj), node) {
		data_mapping_maj.push_back(node);
	endfor }
	svm_refinement<SVC>::uncoarse_SV(graph_min, data_mapping_min, full_summary.SV_min, data_mapping_min);
	svm_refinement<SVC>::uncoarse_SV(graph_maj, data_mapping_maj, full_summary.SV_maj, data_mapping_maj);
	}

	std::cout << "exporting" << std::endl;

	graph_io::writeGraphGDF(graph_min, graph_maj, "graph_testcase_full.gdf");



	/*
	// uncoarsening
        while ( !min_hierarchy.isEmpty() && !maj_hierarchy.isEmpty()) {
		graph_access *G_min = &graph_min;
		graph_access *G_maj = &graph_maj;
		if (!min_hierarchy.isEmpty() && min_hierarchy.size() >= maj_hierarchy.size()) {
			G_min = min_hierarchy.pop_finer_and_project();
		}
		if (!maj_hierarchy.isEmpty()) {
			G_maj = maj_hierarchy.pop_finer_and_project();
		}

		std::cout << "levels - min: " << min_hierarchy.size() << " maj: " << maj_hierarchy.size() << std::endl;

		int level = std::max(min_hierarchy.size(), maj_hierarchy.size());
		std::ostringstream out_graph;
		out_graph << "graph_testcase_" << level << ".gdf";
		std::cout << "exporting " << out_graph.str() << std::endl;
		graph_io::writeGraphGDF(*G_min, *G_maj, out_graph.str());
	}
	//*/
}
