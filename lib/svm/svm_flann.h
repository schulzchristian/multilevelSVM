#ifndef SVM_FLANN_H
#define SVM_FLANN_H

#include "definitions.h"
#include <vector>

class svm_flann
{
public:
        static void run_flann(const std::vector<FeatureVec> & data,
                              std::vector<std::vector<Edge>> & edges,
                              int num_nn = 10);
};


#endif /* SVM_FLANN_H */
