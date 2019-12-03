#include <algorithm>
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
				      PartitionConfig conf,
				      bayesopt::BOptState state)
	: svm_refinement<T>(min_hierarchy, maj_hierarchy, initial_result, conf)
	, opt_state(state) {
	this->seed = conf.seed;
	this->fix_num_vert_stop = conf.fix_num_vert_stop;
	this->bayes_max_steps = conf.bayes_max_steps;
}

template<class T>
bayes_refinement<T>::~bayes_refinement() {}

template<class T>
svm_result<T> bayes_refinement<T>::step(const svm_data & min_sample, const svm_data & maj_sample) {
	std::cout << "BAYES refinement at level " << this->get_level() << std::endl;

        std::vector<NodeID> sv_min = this->result.best().SV_min;
        std::vector<NodeID> sv_maj = this->result.best().SV_maj;
        this->uncoarse(sv_min, sv_maj);

        std::cout << "current level nodes"
                  << " min " << this->uncoarsed_data_min.size()
                  << " maj " << this->uncoarsed_data_maj.size()
                  << std::endl;

        svm_instance instance;
        instance.read_problem(this->uncoarsed_data_min, this->uncoarsed_data_maj);
	std::unique_ptr<svm_solver<T>> solver = svm_solver_factory::create<T>(instance);

        // if (this->uncoarsed_data_min.size() + this->uncoarsed_data_maj.size() < this->num_skip_ms) {
	size_t data_size = this->uncoarsed_data_min.size() + this->uncoarsed_data_maj.size();
	size_t max_init_size = 2 * this->fix_num_vert_stop;
	float clipped_size = std::min(std::max(data_size, max_init_size), (size_t) this->num_skip_ms);
	float ratio = (clipped_size - max_init_size) / (this->num_skip_ms - max_init_size);
	int iterations = std::round( (1 - ratio) * 5);


	if (!this->training_inherit) {
		//reset optimization state when only uncontracting one class
		this->opt_state = bayesopt::BOptState();
	}
	this->result = train_bayes(*solver, min_sample, maj_sample,
				   this->opt_state, iterations, this->seed);
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
					       BOptState & state,
					       int optimization_steps,
					       long seed) {
	std::vector<svm_summary<T>> summaries;
	
	Parameters params;
	params.n_iterations = 10;
	params.n_iter_relearn = 1;
	params.l_type = L_MCMC;
	params.random_seed = seed;
	params.verbose_level = -1;
	params.noise = 0.03;
	params.force_jump = 5;
	SolverOptimization<T> optimizer(params, solver, min_sample, maj_sample, summaries);

	boost::numeric::ublas::vector<double> bestPoint(2);
	boost::numeric::ublas::vector<double> lowerBound(2);
	boost::numeric::ublas::vector<double> upperBound(2);
	// C range
	lowerBound[0] = -15;
	upperBound[0] = 15;
	// gamma range
	lowerBound[1] = -10;
	upperBound[1] = 10;
	optimizer.setBoundingBox(lowerBound,upperBound);


	if (state.mX.size() > 0) {
		// restore previous optimization
		optimizer.restoreOptimization(state);
		bestPoint = optimizer.getFinalResult();
		// refinement optimization
		optimizer.forceOptimization(bestPoint);
		std::cout << optimization_steps << "\n";
		for (int i = 0; i < optimization_steps; ++i) {
			optimizer.stepOptimization();
		}
		optimizer.getFinalResult();
	} else {
		//Define bounds and optimize
		optimizer.optimize(bestPoint);
	}
	optimizer.saveOptimization(state);

	svm_result<T> result(summaries, solver.get_instance());

	result.best().print();

	return result;
}


template class bayes_refinement<svm_model>;
template class bayes_refinement<SVC>;
