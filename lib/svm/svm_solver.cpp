#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <functional>

#include "svm_solver.h"
#include "svm_convert.h"
#include "param_search.h"
#include "timer.h"

svm_solver::svm_solver(const svm_instance & instance)
    : instance(instance) {
        this->param.svm_type = C_SVC;
        this->param.kernel_type = RBF;
        this->param.degree = 3;
        this->param.gamma = 2;	// doc suggests 1/num_features
        this->param.coef0 = 0;
        this->param.nu = 0.5;
        this->param.cache_size = 100;
        this->param.C = 32;
        this->param.eps = 1e-3;
        this->param.p = 0.1;
        this->param.shrinking = 1;
        this->param.probability = 0;
        this->param.nr_weight = 0;
        this->param.weight_label = NULL;
        this->param.weight = NULL;
}

svm_solver::svm_solver() {
}

void svm_solver::train() {
        svm_problem prob;
        prob.l = this->instance.size();
        prob.y = this->instance.label_data();
        prob.x = this->instance.node_data();

        const char * error_msg = svm_check_parameter(&prob, &(this->param));
        if (error_msg != NULL) {
                std::cout << error_msg << std::endl;
                std::cout << "we are exiting due to bad parameters"  << std::endl;
                exit(0);
        }

        svm_model * trained_model = svm_train(&prob, &(this->param));

        this->model = std::shared_ptr<svm_model>
            (trained_model, [](svm_model* m) { svm_free_and_destroy_model(&m); });
}

svm_result svm_solver::train_initial(const svm_data & min_sample, const svm_data & maj_sample) {
        svm_result result;
        std::vector<svm_param> params;

        // first search
        std::cout << "1st sweep with initial logC=0 logGamma=0" << std::endl;
        params = param_search::ud(-5, 15, -10, 10, true);

        result = train_range(params, min_sample, maj_sample);
        svm_summary good = result.best();

        std::cout << "2nd sweep with logC=" << good.C_log << " logGamma=" << good.gamma_log << std::endl;
        good.print();

        // second search
        params = param_search::ud(-5, 15, -10, 10, false, true, good.C_log, good.gamma_log);
        params.pop_back();

        svm_result second_res = train_range(params, min_sample, maj_sample);
        second_res.add(result);
        svm_summary best = second_res.best();

        std::cout << "BEST (" << best.C_log << "," << best.gamma_log << ")"<< std::endl;
        best.print();

        // train this solver to the best found parameters
        this->param.C = best.C;
        this->param.gamma = best.gamma;
        this->train();

        return second_res;
}

svm_result svm_solver::train_refinement(const svm_data & min_sample, const svm_data & maj_sample,
                                        bool inherit_ud, float param_c, float param_g) {
        svm_result result;
        std::vector<svm_param> params;
        if (!inherit_ud) {
                // first search
                std::cout << "1st sweep with logC=" << param_c << " logGamma="<< param_g << std::endl;
                params = param_search::ud(-5, 15, -10, 10, true, true, param_c, param_g);

                result = train_range(params, min_sample, maj_sample);
                svm_summary good = result.best();

                std::cout << "2nd sweep with logC=" << good.C_log << " logGamma=" << good.gamma_log << std::endl;
                good.print();

                // second search
                params = param_search::ud(-5, 15, -10, 10, false, true, good.C_log, good.gamma_log);
                params.pop_back();
        } else {
                std::cout << "2nd sweep with logC=" << param_c << " logGamma="<< param_g << std::endl;
                params = param_search::ud(-5, 15, -10, 10, false, true, param_c, param_g);
                params.push_back(std::make_pair(param_c, param_g));
        }


        svm_result second_res = train_range(params, min_sample, maj_sample);
        second_res.add(result);
        svm_summary best = second_res.best();

        std::cout << "BEST (" << best.C_log << "," << best.gamma_log << ")"<< std::endl;
        best.print();

        // train this solver to the best found parameters
        this->param.C = best.C;
        this->param.gamma = best.gamma;
        this->train();

        return second_res;
}


svm_result svm_solver::train_range(const std::vector<svm_param> & params,
                                   const svm_data & min_sample,
                                   const svm_data & maj_sample) {
        std::vector<svm_summary> summaries;

        for (auto&& p : params) {
                svm_solver cur_solver(*this); // (copy ctor) use this instances prob and param values

                cur_solver.param.C = pow(2, p.first);
                cur_solver.param.gamma = pow(2, p.second);

                cur_solver.train();

                std::cout << std::setprecision(2)
                          << std::fixed
                          << "log C=" << p.first
                          << "\tlog gamma=" << p.second
                          << std::flush;

                // if (cur_solver.model->l > (cur_solver.instance.num_min + cur_solver.instance.num_maj) * 0.9
                //     && !summaries.empty()) {
                //         // don't evaluate models which are very likely prone to over fitting
                //         // but at least evaluate once
                //         std::cout << "not evaluated " << cur_solver.model->l << std::endl;
                //         continue;
                // }

                svm_summary cur_summary = cur_solver.predict_validation_data(min_sample, maj_sample);

                cur_summary.print_short();

                summaries.push_back(cur_summary);
        }

        return svm_result(summaries, this->instance);
}

int svm_solver::predict(const std::vector<svm_node> & nodes) {
        return svm_predict(this->model.get(), nodes.data());
}

svm_summary svm_solver::predict_validation_data(const svm_data & min, const svm_data & maj) {
        size_t tp = 0, tn = 0, fp = 0, fn = 0;

        for (const auto& instance : min) {
                int res = this->predict(instance);
                if (res == 1) {
                        tp++;
                } else {
                        fn++;
                }
        }

        for (const auto& instance : maj) {
                int res = this->predict(instance);
                if (res == -1) {
                        tn++;
                } else {
                        fp++;
                }
        }

        return svm_summary(*this->model, this->instance, tp, tn, fp, fn);
}

void svm_solver::set_C(float C) {
        this->param.C = C;
}

void svm_solver::set_gamma(float gamma) {
        this->param.gamma = gamma;
}
