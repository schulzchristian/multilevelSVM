#ifndef KFOLD_IMPORT_H
#define KFOLD_IMPORT_H

#include "k_fold.h"
#include "partition/partition_config.h"

class k_fold_import: public k_fold
{
public:
        k_fold_import(const PartitionConfig & config, int num_exp, const std::string & basename);
        virtual ~k_fold_import();

protected:
        virtual void next_intern(double & io_time) override;
	double read_class(const std::string & filename,
			  graph_access & target_graph,
			  std::vector<std::vector<svm_node>> & target_val);

        std::string basename;
        int num_exp;
        int num_nn;
        bool bidirectional;
	float sample_percent;
};

#endif /* KFOLD_IMPORT_H */
