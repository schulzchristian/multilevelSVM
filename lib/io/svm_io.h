#ifndef SVMIO_H_
#define SVMIO_H_

#include <vector>

#include "definitions.h"
#include "svm/svm_definitions.h"


class svm_io {
public:
        static void readFeaturesLines(const std::string & filename, std::vector<FeatureVec> & data);

        static void readTestSplit(const std::string & filename, std::vector<svm_feature> & min_test_data,
                                  std::vector<svm_feature> & maj_test_data);

};

#endif /* SVMIO_H_ */
