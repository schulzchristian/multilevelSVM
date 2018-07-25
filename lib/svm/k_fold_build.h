#ifndef KFOLD_BUILD_H
#define KFOLD_BUILD_H

#include "k_fold.h"

class k_fold_build: public k_fold
{
public:
        k_fold_build(int num_iter, const std::string & basename);
        virtual ~k_fold_build();

protected:
        virtual void next_intern() override;

        void readData(const std::string & filename);
        void calculate_kfold_class(const std::vector<FeatureVec> & features_full,
                                   graph_access & target_graph,
                                   std::vector<std::vector<svm_node>> & target_test);

        std::vector<FeatureVec> min_features;
        std::vector<FeatureVec> maj_features;
};

#endif /* KFOLD_BUILD_H */
