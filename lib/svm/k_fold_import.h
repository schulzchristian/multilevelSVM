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

        std::string basename;
        int num_exp;
        bool bidirectional;
        int num_nn;
};

#endif /* KFOLD_IMPORT_H */
