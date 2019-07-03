#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <functional>
#include <thundersvm/model/svc.h>

#include "svm/param_search.h"
#include "svm/svm_solver.h"
#include "svm/svm_convert.h"
#include "tools/timer.h"

template<class T>
svm_solver<T>::svm_solver(const svm_instance & instance)
    : instance(instance) {
        this->param.svm_type = C_SVC;
        this->param.kernel_type = RBF;
        this->param.degree = 3;
        this->param.gamma = 2;	// doc suggests 1/num_features
        this->param.coef0 = 0;
        this->param.nu = 0.5;
        this->param.cache_size = static_cast<size_t>(8192);
        this->param.C = 32;
        this->param.eps = 1e-3;
        this->param.p = 0.1;
        this->param.shrinking = 1;
        this->param.probability = 0;
        this->param.nr_weight = 0;
        this->param.weight_label = NULL;
        this->param.weight = NULL;
}

template<class T>
svm_solver<T>::svm_solver() {
}

template<class T>
svm_result<T> svm_solver<T>::train_grid(const svm_data & min_sample, const svm_data & maj_sample) {
        svm_result<T> result;
        std::vector<svm_param> params;

        // first search
        std::cout << "Grid search C:-5:15:2 gamaa:3:-15:-2" << std::endl;
        params = param_search::grid(-5, 15, 2, 3, -15, -2);

        result = train_range(params, min_sample, maj_sample);
        svm_summary<T> good = result.best();

        // train this solver to the best found parameters
        this->param.C = good.C;
        this->param.gamma = good.gamma;
        this->model = good.model;

        return result;
}

template<class T>
svm_result<T> svm_solver<T>::train_range(const std::vector<svm_param> & params,
					 const svm_data & min_sample,
					 const svm_data & maj_sample) {
	std::vector<svm_summary<T>> summaries;

        for (auto&& p : params) {
		// (copy ctor) use this instances prob and param values
                // svm_solver<T> cur_solver(*this);

		timer t;

                param.C = pow(2, p.first);
                param.gamma = pow(2, p.second);

                std::cout << std::setprecision(2)
                          << std::fixed
                          << "log C=" << std::setw(6) << p.first
                          << "\tlog gamma=" << std::setw(6) << p.second
			  << std::flush;

                this->train();

		std::cout << "\ttime=" << t.elapsed()
                          << std::flush;

                // if (cur_solver.model->l > (cur_solver.instance.num_min + cur_solver.instance.num_maj) * 0.9
                //     && !summaries.empty()) {
                //         // don't evaluate models which are very likely prone to over fitting
                //         // but at least evaluate once
                //         std::cout << "not evaluated " << cur_solver.model->l << std::endl;
                //         continue;
                // }

                svm_summary<T> cur_summary = this->build_summary(min_sample, maj_sample);
                cur_summary.print_short();
                summaries.push_back(cur_summary);
        }

        return svm_result<T>(summaries, this->instance);
}

template<class T>
svm_summary<T> svm_solver<T>::build_summary(const svm_data & min, const svm_data & maj) {
        size_t tp = 0, tn = 0, fp = 0, fn = 0;

        for (int res : this->predict_batch(min)) {
                if (res == 1) {
                        tp++;
                } else {
                        fn++;
                }
        }

        for (int res : this->predict_batch(maj)) {
                if (res == -1) {
                        tn++;
                } else {
                        fp++;
                }
        }

	auto SV_pair = this->get_SV();

	svm_summary<T> summary(tp, tn, fp, fn);

        summary.model = model;
	summary.SV_min = SV_pair.first;
	summary.SV_maj = SV_pair.second;
	summary.C = this->param.C;
	summary.gamma = this->param.gamma;
        summary.C_log = std::log(this->param.C) / std::log(2);
        summary.gamma_log = std::log(this->param.gamma) / std::log(2);

        return summary;
}

template<class T>
std::vector<int> svm_solver<T>::predict_batch(const svm_data & data) {
	std::vector<int> result;
	result.reserve(data.size());
	for (const auto& instance : data) {
		result.push_back(this->predict(instance));
	}
	return result;
}

template<class T>
void svm_solver<T>::set_C(float C) {
        this->param.C = C;
}

template<class T>
void svm_solver<T>::set_gamma(float gamma) {
        this->param.gamma = gamma;
}

template<class T>
void svm_solver<T>::set_model(std::shared_ptr<T> new_model) {
	this->model = new_model;
}

template class svm_solver<svm_model>;
template class svm_solver<SVC>;
