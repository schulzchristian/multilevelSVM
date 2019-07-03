#include <iostream>
#include <unordered_set>
#include <thundersvm/model/svc.h>

#include "svm/svm_refinement.h"
#include "svm/svm_convert.h"
#include "tools/timer.h"


template<class T>
svm_refinement<T>::svm_refinement(graph_hierarchy & min_hierarchy,
				  graph_hierarchy & maj_hierarchy,
				  const svm_result<T> & initial_result,
				  PartitionConfig conf)
    : result(initial_result) {
        this->min_hierarchy = &min_hierarchy;
        this->maj_hierarchy = &maj_hierarchy;
	this->G_min = min_hierarchy.get_coarsest();
	this->G_maj = maj_hierarchy.get_coarsest();
        this->uncoarsed_data_min = svm_convert::graph_to_nodes(* this->min_hierarchy->get_coarsest());
        this->uncoarsed_data_maj = svm_convert::graph_to_nodes(* this->maj_hierarchy->get_coarsest());
        this->training_inherit = false;
        this->num_skip_ms = conf.num_skip_ms;
}

template<class T>
svm_refinement<T>::~svm_refinement() {
}

template<class T>
bool svm_refinement<T>::is_done() {
        return min_hierarchy->isEmpty() && maj_hierarchy->isEmpty();
}

template<class T>
int svm_refinement<T>::get_level() {
        return std::max(min_hierarchy->size(), maj_hierarchy->size());
}

template<class T>
void svm_refinement<T>::uncoarse() {
        std::vector<NodeID> sv_min = result.best().SV_min;
        std::vector<NodeID> sv_maj = result.best().SV_maj;

        // if maj_hierarchy is larger then start by only uncoarse the maj graph
        if (!min_hierarchy->isEmpty() && min_hierarchy->size() >= maj_hierarchy->size()) {
                std::cout << "minority uncoarsed" << std::endl;
                this->G_min = min_hierarchy->pop_finer_and_project();
                CoarseMapping* mapping_min = min_hierarchy->get_mapping_of_current_finer();
                this->uncoarsed_data_min = uncoarse_graph(*G_min, *mapping_min, sv_min);
                this->training_inherit = true; // after the first uncoarsening of the min data inherit params
        }
        if (!maj_hierarchy->isEmpty()) {
                std::cout << "majority uncoarsed" << std::endl;
                this->G_maj = maj_hierarchy->pop_finer_and_project();
                CoarseMapping* mapping_maj = maj_hierarchy->get_mapping_of_current_finer();
                this->uncoarsed_data_maj = uncoarse_graph(*G_maj, *mapping_maj, sv_maj);
        }
}

template<class T>
svm_data svm_refinement<T>::uncoarse_graph(const graph_access & G,
					   const CoarseMapping & coarse_mapping,
					   const std::vector<NodeID> & sv) {
	return this->get_SV_neighbors(G, coarse_mapping, sv);
}

template<class T>
svm_data svm_refinement<T>::get_SV_neighbors(const graph_access & G,
					     const CoarseMapping & coarse_mapping,
					     const std::vector<NodeID> & sv) {
	svm_data neighbors;
	std::unordered_set<NodeID> sv_set{sv.begin(), sv.end()};

        forall_nodes(G, node) {
                NodeID coarse_node = coarse_mapping[node];
                if (sv_set.find(coarse_node) != sv_set.end()) {
                        svm_feature feature = svm_convert::feature_to_node(G.getFeatureVec(node));
                        neighbors.push_back(std::move(feature));
                }
        } endfor

        std::cout << "uncoarsened nodes " << G.number_of_nodes()
                  << " SV " << sv.size()
                  << " resulting neighbors " << neighbors.size()
                  << std::endl;

        return neighbors;
}


template class svm_refinement<svm_model>;
template class svm_refinement<SVC>;
