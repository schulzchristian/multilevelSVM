#include "k_fold_build.h"
#include "io/graph_io.h"
#include "io/svm_io.h"
#include "svm/svm_convert.h"
#include "svm/svm_flann.h"
#include "tools/random_functions.h"
#include "tools/timer.h"


k_fold_build::k_fold_build(const PartitionConfig & config, const std::string & filename)
        : k_fold(config) {
        this->num_nn = config.num_nn;
        this->bidirectional = config.bidirectional;
	this->sample_percent  = config.sample_percent;
        readData(filename);
}

k_fold_build::~k_fold_build() {
}

void k_fold_build::readData(const std::string & filename) {
        timer t;

        svm_io::readFeaturesLines(filename + "_min_data", this->min_features);
        svm_io::readFeaturesLines(filename + "_maj_data", this->maj_features);

	if (this->sample_percent < 1 - 0.0001f) {
		this->min_features = svm_io::take_sample(this->min_features, this->sample_percent);
		this->maj_features = svm_io::take_sample(this->maj_features, this->sample_percent);
	}

	random_functions::permutate_vector_good(this->min_features, false);
        random_functions::permutate_vector_good(this->maj_features, false);

        std::cout << "io time: " << t.elapsed() << std::endl;

        std::cout << "full graph -"
                  << " min: " << this->min_features.size()
                  << " maj: " << this->maj_features.size()
                  << " features: " << this->min_features[0].size() << std::endl;
}


void k_fold_build::next_intern(double & io_time) {
        this->cur_min_train.clear();
        this->cur_maj_train.clear();
        this->cur_min_val.clear();
        this->cur_maj_val.clear();
        this->cur_min_test.clear();
        this->cur_maj_test.clear();

        calculate_kfold_class(this->min_features, this->cur_min_graph, this->cur_min_val, this->cur_min_test);
        calculate_kfold_class(this->maj_features, this->cur_maj_graph, this->cur_maj_val, this->cur_maj_test);
}

void k_fold_build::calculate_kfold_class(const std::vector<FeatureVec> & features_full,
					 graph_access & target_graph,
					 std::vector<std::vector<svm_node>> & target_val,
					 std::vector<std::vector<svm_node>> & target_test) {
	NodeID nodes          = features_full.size();
        NodeID test_size      = floor(nodes / this->iterations);
        NodeID test_start     = k_fold::cur_iteration * test_size;
        NodeID test_end       = (k_fold::cur_iteration + 1) * test_size;
        NodeID train_size     = nodes - test_size;
        NodeID val_size       = floor(train_size * this->validation_percent);
	NodeID val_start;
	NodeID val_end;
	if (test_start > val_size) {
		val_start = test_start - val_size;
		val_end = test_start;
	} else {
		val_start = test_end;
		val_end = test_end + val_size;
	}

        std::vector<FeatureVec> feature_subset(features_full);

	if (this->validation_seperate) {
		feature_subset.erase(feature_subset.begin() + std::min(val_start, test_start),
				     feature_subset.begin() + std::max(val_end, test_end));
	} else {
		feature_subset.erase(feature_subset.begin() + test_start,
				     feature_subset.begin() + test_end);
	}

	// prepare graph
        std::vector<std::vector<Edge>> edges_subset;
        svm_flann::run_flann(feature_subset, edges_subset, this->num_nn);

        EdgeID edges;
        edges = nodes * this->num_nn;
        if (bidirectional) {
                edges = graph_io::makeEdgesBidirectional(edges_subset);
        }
        graph_io::readGraphFromVec(target_graph, edges_subset, edges*2);
        graph_io::readFeatures(target_graph, feature_subset);

	// build validation set
        std::vector<FeatureVec> val_subset = std::vector<FeatureVec>();
        val_subset.insert(val_subset.end(),
			  features_full.begin() + val_start,
			  features_full.begin() + val_end);

        target_val.reserve(val_size);
        for (const FeatureVec & f : val_subset) {
                target_val.push_back(svm_convert::feature_to_node(f));
        }

	// build test set
        std::vector<FeatureVec> test_subset = std::vector<FeatureVec>();
        test_subset.insert(test_subset.end(),
			   features_full.begin() + test_start,
			   features_full.begin() + test_end);

        target_test.reserve(test_size);
        for (const FeatureVec & f : test_subset) {
                target_test.push_back(svm_convert::feature_to_node(f));
        }
}
