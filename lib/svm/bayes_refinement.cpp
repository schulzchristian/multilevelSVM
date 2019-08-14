#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <thundersvm/model/svc.h>
#include <svm.h>

#include "svm/bayes_refinement.h"
#include "svm/param_search.h"
#include "svm/svm_solver_factory.h"
#include "svm/svm_instance.h"

template<class T>
bayes_refinement<T>::bayes_refinement(graph_hierarchy & min_hierarchy,
				graph_hierarchy & maj_hierarchy,
				const svm_result<T> & initial_result,
				PartitionConfig conf)
	: svm_refinement<T>(min_hierarchy, maj_hierarchy, initial_result, conf) {
	this->seed = conf.seed;
}

template<class T>
bayes_refinement<T>::~bayes_refinement() {}

template<class T>
svm_result<T> bayes_refinement<T>::step(const svm_data & min_sample, const svm_data & maj_sample) {
	std::cout << "BAYES refinement at level " << this->get_level() << std::endl;

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
                        this->result = this->train_bayes(*solver, min_sample, maj_sample, this->seed);
                        // this->result = train_refinement(*solver, min_sample, maj_sample,
			// 				this->inherit_bayes,
			// 				this->result.best().C_log,
			// 				this->result.best().gamma_log);
		} else {
			// uncoarsend just a single class so to parameter training again
                        this->result = train_bayes(*solver, min_sample, maj_sample, this->seed);
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


using namespace bayesopt;

template<class T>
class SolverOptimization: public ContinuousModel
{
public:
	SolverOptimization(Parameters param, svm_solver<T> & solver,
			   const svm_data & min_sample, const svm_data & maj_sample,
			   std::vector<svm_summary<T>> & summaries)
		: ContinuousModel(2, param),
		  solver(solver),
		  min_sample(min_sample),
		  maj_sample(maj_sample),
		  summaries(summaries) {
	}

	double evaluateSample(const boost::numeric::ublas::vector<double> &query) {
		svm_param p = std::make_pair(query[0], query[1]);
		auto summary = solver.train_single(p, min_sample, maj_sample);
		summaries.push_back(summary);
		return summary.eval(solver.get_instance());
	}

	bool checkReachability(const boost::numeric::ublas::vector<double> &query) { 
		return true;
	}

	svm_solver<T> & solver;
	const svm_data & min_sample;
	const svm_data & maj_sample;
	std::vector<svm_summary<T>> & summaries;
};

template<class T>
svm_result<T> bayes_refinement<T>::train_bayes(svm_solver<T> & solver,
					       const svm_data & min_sample,
					       const svm_data & maj_sample,
					       long seed) {
	std::vector<svm_summary<T>> summaries;
	
	Parameters params;
	params.n_iterations = 10;
	params.n_iter_relearn = 1;
	params.l_type = L_MCMC;
	params.random_seed = seed;
	SolverOptimization<T> optimizer(params, solver, min_sample, maj_sample, summaries);

	//Define bounds and prepare result.
	boost::numeric::ublas::vector<double> bestPoint(2);
	boost::numeric::ublas::vector<double> lowerBound(2);
	boost::numeric::ublas::vector<double> upperBound(2);
	// C range
	lowerBound[0] = -10;
	upperBound[0] = 10;
	// gamma range
	lowerBound[1] = -10;
	upperBound[1] = 10;
	optimizer.setBoundingBox(lowerBound,upperBound);

	optimizer.optimize(bestPoint);

	svm_result<T> result(summaries, solver.get_instance());

	return result;
}


template class bayes_refinement<svm_model>;
template class bayes_refinement<SVC>;
