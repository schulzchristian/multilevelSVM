#ifndef SVM_SOLVER_FACTORY_H
#define SVM_SOLVER_FACTORY_H

#include <svm.h>
#include <thundersvm/model/svc.h>
#include <memory>

#include "svm/svm_solver.h"
#include "svm/svm_solver_libsvm.h"
#include "svm/svm_solver_thunder.h"
#include "svm/svm_instance.h"

class svm_solver_factory {
public:
	template<class T>
	static std::unique_ptr<svm_solver<T>> create(const svm_instance & instance);
};

template<> inline
std::unique_ptr<svm_solver<svm_model>> svm_solver_factory::create(const svm_instance & instance) {
	return std::make_unique<svm_solver_libsvm>(instance);
}

template<> inline
std::unique_ptr<svm_solver<SVC>> svm_solver_factory::create(const svm_instance & instance) {
	return std::make_unique<svm_solver_thunder>(instance);
}

#endif /* SVM_SOLVER_FACTORY_H */
