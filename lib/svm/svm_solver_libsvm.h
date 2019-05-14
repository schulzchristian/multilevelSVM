#ifndef SVM_SOLVER_LIBSVM_H
#define SVM_SOLVER_LIBSVM_H

#include <utility>
#include <svm.h>

#include "svm_solver.h"

class svm_solver_libsvm : public svm_solver<svm_model>
{
public:
	svm_solver_libsvm();
        svm_solver_libsvm(const svm_instance & instance);

        void train() override;
        int predict(const std::vector<svm_node> & node) override;
	std::pair<std::vector<NodeID>, std::vector<NodeID>> get_SV() override;
};

#endif /* SVM_SOLVER_LIBSVM_H */
