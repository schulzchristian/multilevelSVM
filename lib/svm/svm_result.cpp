#include <algorithm>

#include "svm_result.h"

svm_result::svm_result(const std::vector<svm_summary> & summaries, const svm_instance & instance)
        : summaries(summaries), instance(instance) {

        this->sort_summaries();
}

svm_summary svm_result::best() {
        for (size_t i = 0; i < summaries.size(); ++i) {
                if(summaries[i].Gmean > 0.05) {
                        return summaries[i];
                }
        }
        // in case there is no model with gmean larger than zero, return the 1st one
        return summaries[0];
}

void svm_result::add(const svm_result & result) {
        this->add(result.summaries);
}

void svm_result::add(const std::vector<svm_summary> & to_add) {
        this->summaries.insert(this->summaries.end(), to_add.begin(), to_add.end());

        this->sort_summaries();
}

std::vector<svm_param> svm_result::all_params() {
        std::vector<svm_param> seq;

        for (const svm_summary & smry : this->summaries) {
                seq.push_back(std::make_pair(smry.C_log, smry.gamma_log));
        }

        return seq;
}

void svm_result::sort_summaries() {
        std::sort(this->summaries.begin(), this->summaries.end(), summary_cmp_better_gmean_sv::comp);
}

static size_t svm_result::get_best_index(const std::vector<std::pair<svm_summary,svm_instance>> vec) {
        size_t best_index = 0;

        for (size_t i = 1; i < vec.size(); i++) {
                if (summary_cmp_better_gmean::comp(vec[i].first, vec[best_index].first)) {
                        best_index = i;
                }
        }
        return best_index;
}
