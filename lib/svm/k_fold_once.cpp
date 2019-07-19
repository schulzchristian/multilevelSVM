#include "k_fold_once.h"

k_fold_once::k_fold_once(const PartitionConfig & config, const std::string & filename)
	: k_fold_build(config, filename) {
}

k_fold_once::~k_fold_once() {
}

void k_fold_once::next_intern(double & io_time) {
	k_fold_build::next_intern(io_time);
	this->iterations = 1;
}
