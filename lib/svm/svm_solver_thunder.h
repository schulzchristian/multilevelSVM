#ifndef SVM_SOLVER_THUNDER_H
#define SVM_SOLVER_THUNDER_H

#include <utility>
#include <thundersvm/model/svc.h>

#include "svm_solver.h"

class svm_solver_thunder : public svm_solver<SVC>
{
public:
	svm_solver_thunder();
        svm_solver_thunder(const svm_instance & instance);

        void train() override;
        int predict(const std::vector<svm_node> & node) override;
	std::vector<int> predict_batch(const svm_data & data) override;
	std::pair<std::vector<NodeID>, std::vector<NodeID>> get_SV() override;
};

#endif /* SVM_SOLVER_THUNDER_H */
