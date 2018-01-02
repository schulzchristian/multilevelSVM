#ifndef SVM_DEFINITIONS_H
#define SVM_DEFINITIONS_H

#include <vector>
#include <utility>
#include <svm.h>

#include "svm_summary.h"

typedef std::vector<svm_node> svm_feature;
typedef std::vector<svm_feature> svm_data;
typedef std::pair<float,float> svm_para;
typedef std::vector<svm_summary> svm_result;

#endif /* SVM_DEFINITIONS_H */
