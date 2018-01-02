#ifndef SVM_DESC_H
#define SVM_DESC_H

#include "definitions.h"

struct svm_desc
{
        NodeID num_min;
        NodeID num_maj;
        size_t features;
};

#endif /* SVM_DESC_H */
