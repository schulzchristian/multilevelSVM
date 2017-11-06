#ifndef SVM_SOLVER_H
#define SVM_SOLVER_H

#include <svm.h>
#include "data_structure/graph_access.h"
#include "timer.h"

class svm_solver
{
public:
        svm_solver();
        virtual ~svm_solver();

        void read_problem(const graph_access & G_maj, const graph_access & G_min);
        void train();
        int predict(const FeatureVec & vec);

private:
        void add_graph_to_problem(const graph_access & G, int label, NodeID offset);

        svm_problem prob;
        svm_parameter param;
        svm_model *model;

};


#endif /* SVM_SOLVER_H */
