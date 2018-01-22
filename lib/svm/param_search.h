#ifndef PARAM_SEARCH_H
#define PARAM_SEARCH_H

#include <vector>
#include <utility>

#include "svm_definitions.h"
#include "svm_result.h"

class param_search
{
public:
        static std::vector<svm_param> grid(float c_from, float c_to, float c_step,
                                                  float g_from, float g_to, float g_step);

        static std::vector<svm_param> around(float c_center, float c_range, float c_step,
                                             float g_center, float g_range, float g_step);

        static std::vector<svm_param> mlsvm_method(float c_from, float c_to, float g_from, float g_to,
                                                   bool step1, bool inherit = false,
                                                   float param_c = 0, float param_g = 0);

private:
        template<typename T>
        static std::vector<T> range(T from, T to, T step);
};

template<typename T>
std::vector<T> param_search::range(T from, T to, T step) {
        std::vector<T> seq;
        while (true) {
                if (step > 0 && from > to) break;
                if (step < 0 && from < to) break;
                seq.push_back(from);
                from += step;
        }
        return seq;
}


#endif /* PARAM_SEARCH_H */
