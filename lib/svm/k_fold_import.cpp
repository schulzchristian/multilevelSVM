#include "k_fold_import.h"
#include "io/graph_io.h"
#include "io/svm_io.h"
#include "svm/svm_convert.h"
#include "svm/svm_flann.h"
#include "tools/random_functions.h"
#include "tools/timer.h"


k_fold_import::k_fold_import(const PartitionConfig & config, int num_exp, const std::string & basename)
        : k_fold(config) {
        this->num_exp = num_exp;
        this->basename = basename;
        this->bidirectional = config.bidirectional;
        this->num_nn = config.num_nn;
}

k_fold_import::~k_fold_import() {
}

void k_fold_import::next_intern(double & io_time) {
        // TODO don't build the path with hard coded strings, use a path template as argument
        std::string min_train_name = this->basename + "kfold_p_train_data_exp_" + std::to_string(this->num_exp) + "_fold_" + std::to_string(k_fold::cur_iteration) + "_exp_0.1_data";
        std::string maj_train_name = this->basename + "kfold_n_train_data_exp_" + std::to_string(this->num_exp) + "_fold_" + std::to_string(k_fold::cur_iteration) + "_exp_0.1_data";
        std::string test_name = this->basename + "kfold_test_data_exp_" + std::to_string(this->num_exp) + "_fold_" + std::to_string(k_fold::cur_iteration) + "_exp_0.1_data";

	io_time += read_class(min_train_name, this->cur_min_graph, this->cur_min_val);
	io_time += read_class(maj_train_name, this->cur_maj_graph, this->cur_maj_val);

	//read test
        std::cout << "reading " << test_name << std::endl;

        timer t;
        svm_io::readTestSplit(test_name, this->cur_min_test, this->cur_maj_test);
        io_time += t.elapsed();
}

double k_fold_import::read_class(const std::string & filename,
				 graph_access & target_graph,
				 std::vector<std::vector<svm_node>> & target_val) {
	double time;
        timer t;
        std::cout << "reading " << filename << std::endl;


        std::vector<FeatureVec> features_full;
        svm_io::readFeaturesLines(filename, features_full);
        time += t.elapsed();

        NodeID val_size = floor(features_full.size() * this->validation_percent);

        std::vector<FeatureVec> feature_subset(features_full);

	if (this->validation_seperate) {
		feature_subset.erase(feature_subset.end() - val_size,
				     feature_subset.end());
	}

	// build graph
        std::vector<std::vector<Edge>> edges_subset;
        svm_flann::run_flann(feature_subset, edges_subset, num_nn);

        EdgeID edges = feature_subset.size() * num_nn * 2;
        if (bidirectional) {
                edges = graph_io::makeEdgesBidirectional(edges_subset);
        }
        graph_io::readGraphFromVec(target_graph, edges_subset, edges * 2);
        graph_io::readFeatures(target_graph, feature_subset);

	// build validation set
        std::vector<FeatureVec> val_subset = std::vector<FeatureVec>();
        val_subset.insert(val_subset.end(),
			  features_full.end() - val_size,
			  features_full.end());

        target_val.reserve(val_size);
        for (const FeatureVec & f : val_subset) {
                target_val.push_back(svm_convert::feature_to_node(f));
        }

	return time;
}

