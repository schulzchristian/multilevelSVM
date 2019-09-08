#include "partition/partition_config.h"

#include <omp.h>

#include "tools/random_functions.h"

void PartitionConfig::print() {
	std::cout << "file: " << this->filename << std::endl;
	std::cout << "num_experiments: " << this->num_experiments << std::endl;
	std::cout << "kfold_iterations: " << this->kfold_iterations << std::endl;
	std::cout << "sample_percent: " << this->sample_percent << std::endl;
	std::cout << "validation_type: " << this->validation_type << std::endl;
	std::cout << "validation_percent: " << this->validation_percent << std::endl;
	std::cout << "validation_seperate: " << this->validation_seperate << std::endl;
	std::cout << "stop rule: " << this->stop_rule << std::endl;
	std::cout << "matching type: " << this->matching_type << std::endl;
	std::cout << "bidirectional: " << this->bidirectional << std::endl;
	std::cout << "cluster_upperbound: " << this->cluster_upperbound << std::endl;
	std::cout << "upper_bound_partition: " << this->upper_bound_partition << std::endl;
	std::cout << "label_iterations: " << this->label_iterations << std::endl;
	std::cout << "node_ordering: " << this->node_ordering << std::endl;
	std::cout << "diameter_upperbound: " << this->diameter_upperbound << std::endl;
	std::cout << "refinement_type: " << this->refinement_type << std::endl;
	std::cout << "num_skip_ms: " << this->num_skip_ms << std::endl;
	std::cout << "inherit_ud: " << this->inherit_ud << std::endl;
	std::cout << "timeout: " << this->timeout << std::endl;
	std::cout << "cores: " << this->n_cores << std::endl;
	std::cout << "seed: " << this->seed << std::endl;
}

void PartitionConfig::apply() {
	random_functions::setSeed(this->seed);
	if (this->n_cores > 0) {
		omp_set_num_threads(this->n_cores);
	}
}
