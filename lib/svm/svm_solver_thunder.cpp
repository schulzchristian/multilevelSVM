#include <algorithm>
#include <functional>

#include "svm/param_search.h"
#include "svm/svm_solver_thunder.h"
#include "svm/svm_convert.h"
#include "tools/timer.h"

svm_solver_thunder::svm_solver_thunder(const svm_instance & instance)
	: svm_solver(instance) {
}

svm_solver_thunder::svm_solver_thunder() : svm_solver() {
}

void svm_solver_thunder::train() {
	this->model = std::shared_ptr<SVC>(new SVC());
	SvmParam param;
	param.svm_type = SvmParam::SVM_TYPE::C_SVC;
	param.kernel_type = SvmParam::KERNEL_TYPE::RBF;
	param.C = this->param.C;
	param.gamma = this->param.gamma;
	param.p = this->param.p;
	param.nu = this->param.nu;
	param.epsilon = this->param.eps;
	param.degree = this->param.degree;
	param.coef0 = this->param.coef0;
	param.nr_weight = this->param.nr_weight;
	param.weight_label = this->param.weight_label;
	param.weight = this->param.weight;
	param.probability = this->param.probability;
	param.max_mem_size = this->param.cache_size * (1 << 20); //MB to Byte
	// param.max_mem_size = -1; // no limit

	DataSet::node2d nodes = instance.node_data_thunder();
	DataSet dataset(nodes, this->instance.features, *this->instance.labels);
	model->train(dataset, param);
}


std::vector<int> svm_solver_thunder::predict_batch(const svm_data & data) {
	DataSet::node2d dataset = svm_convert::svmdata_to_dataset(data);
	//TODO don't use fixed batch size
	std::vector<double> dRes = this->model->predict(dataset, -1); //define batch size
	std::vector<int> iRes(dRes.begin(), dRes.end());
	return iRes;
}

int svm_solver_thunder::predict(const std::vector<svm_node> & nodes) {
	//TODO doesn't work but predict_batch is used anyway
	exit(1);
	DataSet::node2d dataset(1);
	dataset[0].reserve(nodes.size());
	for (size_t i = 0; i < nodes.size() - 1; i++) {
		svm_node n = nodes[i];
		dataset[0][i] = DataSet::node(n.index, n.value);
	}
	double res = this->model->predict(dataset, 50).front();
        return static_cast<int>(res);
}

std::pair<std::vector<NodeID>, std::vector<NodeID>> svm_solver_thunder::get_SV() {
	const std::vector<int> SV_ind = this->model->get_sv_ind();
	std::vector<NodeID> SV_min;
	std::vector<NodeID> SV_maj;

        for (size_t i = 0; i < SV_ind.size(); i++) {
		int index = SV_ind[i];
		// std::cout << index << std::endl;

                if (index < instance.num_min) {
			SV_min.push_back(index);
		} else {
			SV_maj.push_back(index - instance.num_min);
		}
        }

	return std::make_pair(SV_min, SV_maj);
}
