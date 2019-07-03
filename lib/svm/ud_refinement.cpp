#include "svm/ud_refinement.h"

#include "svm/param_search.h"
#include "svm/svm_solver_factory.h"
#include "svm/svm_instance.h"

template<class T>
ud_refinement<T>::ud_refinement(graph_hierarchy & min_hierarchy,
				graph_hierarchy & maj_hierarchy,
				const svm_result<T> & initial_result,
				PartitionConfig conf)
	: svm_refinement<T>(min_hierarchy, maj_hierarchy, initial_result, conf) {
        this->inherit_ud = conf.inherit_ud;
}

template<class T>
ud_refinement<T>::~ud_refinement() {}

template<class T>
svm_result<T> ud_refinement<T>::step(const svm_data & min_sample, const svm_data & maj_sample) {
	std::cout << "UD refinement at level " << get_level() << std::endl;

        uncoarse();

        std::cout << "current level nodes"
                  << " min " << this->uncoarsed_data_min.size()
                  << " maj " << this->uncoarsed_data_maj.size()
                  << std::endl;

        svm_instance instance;
        instance.read_problem(this->uncoarsed_data_min, this->uncoarsed_data_maj);
	std::unique_ptr<svm_solver<T>> solver = svm_solver_factory::create<T>(instance);

        if (this->uncoarsed_data_min.size() + this->uncoarsed_data_maj.size() < this->num_skip_ms) {
                if (this->training_inherit) {
                        this->result = train_refinement(*solver, min_sample, maj_sample,
							this->inherit_ud,
							this->result.best().C_log,
							this->result.best().gamma_log);
		} else {
			// uncoarsend just a single class so to parameter training again
                        this->result = train_ud(*solver, min_sample, maj_sample);
                }
        } else {
                // std::cout << "test over result range" << std::endl;
                // std::vector<svm_param> refine_range = result.all_params();
                // result = solver.train_range(refine_range, min_sample, maj_sample);

                std::cout << "skip training just use log C=" << this->result.best().C_log
                          << " log gamma=" << this->result.best().gamma_log << std::endl;
                solver->set_C(this->result.best().C);
                solver->set_gamma(this->result.best().gamma);
                solver->train();
                svm_summary<T> s = solver->build_summary(min_sample, maj_sample);
                s.print();
                std::vector<svm_summary<T>> vec;
                vec.push_back(s);
                this->result = svm_result<T>(vec, instance);
        }

        return this->result;
}

template<class T>
svm_result<T> ud_refinement<T>::train_ud(svm_solver<T> & solver, const svm_data & min_sample, const svm_data & maj_sample) {
        svm_result<T> result;
        std::vector<svm_param> params;

        // first search
        std::cout << "1st sweep with initial logC=0 logGamma=0" << std::endl;
        params = param_search::ud(-5, 10, -10, 10, true);

        result = solver.train_range(params, min_sample, maj_sample);
        svm_summary<T> good = result.best();

        std::cout << "2nd sweep with logC=" << good.C_log << " logGamma=" << good.gamma_log << std::endl;
        good.print();

        // second search
        params = param_search::ud(-5, 10, -10, 10, false, true, good.C_log, good.gamma_log);
        params.pop_back();

        svm_result<T> second_res = solver.train_range(params, min_sample, maj_sample);
        second_res.add(result);
        svm_summary<T> best = second_res.best();

        std::cout << "BEST (" << best.C_log << "," << best.gamma_log << ")"<< std::endl;
        best.print();

        // set this solver to the best found solver
        solver.set_C(best.C);
        solver.set_gamma(best.gamma);
        solver.set_model(best.model);

        return second_res;
}

template<class T>
svm_result<T> ud_refinement<T>::train_refinement(svm_solver<T> & solver,
						 const svm_data & min_sample,
						 const svm_data & maj_sample,
						 bool inherit_ud, float param_c, float param_g) {
	svm_result<T> result;
        std::vector<svm_param> params;
        if (!inherit_ud) {
                // first search
                std::cout << "1st sweep with logC=" << param_c << " logGamma="<< param_g << std::endl;
                params = param_search::ud(-5, 10, -10, 10, true, true, param_c, param_g);

                result = solver.train_range(params, min_sample, maj_sample);
                svm_summary<T> good = result.best();

                std::cout << "2nd sweep with logC=" << good.C_log << " logGamma=" << good.gamma_log << std::endl;
                good.print();

                // second search
                params = param_search::ud(-5, 10, -10, 10, false, true, good.C_log, good.gamma_log);
                params.pop_back();
        } else {
                std::cout << "2nd sweep with logC=" << param_c << " logGamma="<< param_g << std::endl;
                params = param_search::ud(-5, 10, -10, 10, false, true, param_c, param_g);
                params.push_back(std::make_pair(param_c, param_g));
        }


        svm_result<T> second_res = solver.train_range(params, min_sample, maj_sample);
        second_res.add(result);
        svm_summary<T> best = second_res.best();

        std::cout << "BEST (" << best.C_log << "," << best.gamma_log << ")"<< std::endl;
        best.print();

        // set this solver to the best found solver
        solver.set_C(best.C);
        solver.set_gamma(best.gamma);
        solver.set_model(best.model);

        return second_res;
}


template class ud_refinement<svm_model>;
template class ud_refinement<SVC>;
