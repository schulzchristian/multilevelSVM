#ifndef SVM_DEFINITIONS_H
#define SVM_DEFINITIONS_H

#include <vector>
#include <utility>
#include <svm.h>

typedef std::vector<svm_node> svm_feature;
typedef std::vector<svm_feature> svm_data;
typedef std::pair<float,float> svm_param;

#endif /* SVM_DEFINITIONS_H */
