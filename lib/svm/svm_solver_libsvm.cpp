#include <algorithm>
#include <functional>

#include "svm/param_search.h"
#include "svm/svm_solver_libsvm.h"
#include "svm/svm_convert.h"
#include "tools/timer.h"

svm_solver_libsvm::svm_solver_libsvm(const svm_instance & instance)
	: svm_solver(instance) {
}

svm_solver_libsvm::svm_solver_libsvm() : svm_solver() {
}

void svm_solver_libsvm::train() {
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

int svm_solver_libsvm::predict(const std::vector<svm_node> & nodes) {
        return svm_predict(this->model.get(), nodes.data());
}

std::pair<std::vector<NodeID>, std::vector<NodeID>> svm_solver_libsvm::get_SV() {
	std::vector<NodeID> SV_min;
	SV_min.reserve(model->nSV[0]);
        for(int i = 0; i < model->nSV[0]; i++) {
                // libSVM uses 1 as first index while we need our NodeID
                NodeID id = model->sv_indices[i] - 1;
                SV_min.push_back(id);
        }

	std::vector<NodeID> SV_maj;
	SV_maj.reserve(model->nSV[1]);
        for(int i = 0; i < model->nSV[1]; i++) {
                int id = model->sv_indices[model->nSV[0] + i] - 1 - instance.num_min;
                SV_maj.push_back(id);
        }

	return std::make_pair(SV_min, SV_maj);
}
