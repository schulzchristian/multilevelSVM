#ifndef SVM_SOLVER_FACTORY_H
#define SVM_SOLVER_FACTORY_H

#include <svm.h>
#include <memory>

#include "svm/svm_solver.h"
#include "svm/svm_solver_libsvm.h"
#include "svm/svm_instance.h"

class svm_solver_factory {
public:
	template<class T>
	static std::unique_ptr<svm_solver<T>> create(const svm_instance & instance);
};

template<>
std::unique_ptr<svm_solver<svm_model>> svm_solver_factory::create(const svm_instance & instance) {
	// TODO: c++14 allows for make_unique
	/* return std::make_unique<svm_solver_libsvm>(instance); */
	return std::unique_ptr<svm_solver_libsvm>(new svm_solver_libsvm(instance));
}

#endif /* SVM_SOLVER_FACTORY_H */
