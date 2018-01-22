#ifndef SVM_RESULT_H
#define SVM_RESULT_H

#include <vector>

#include "definitions.h"
#include "svm_summary.h"
#include "svm_instance.h"

class svm_result
{
public:
        svm_result(const std::vector<svm_summary> & summaries, const svm_instance & instance);

        svm_summary best();

        void sort_summaries();

        void add(const std::vector<svm_summary> & to_add);
        void add(const svm_result & result);

        std::vector<svm_param> all_params();

        std::vector<svm_summary> summaries;
        svm_instance instance;


        static size_t get_best_index(const std::vector<std::pair<svm_summary,svm_instance>> vec);
};


#endif /* SVM_RESULT_H */
