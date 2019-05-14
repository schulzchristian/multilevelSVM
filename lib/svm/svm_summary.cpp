#include <iostream>
#include <iomanip>
#include <cmath>

#include "svm_summary.h"

template<class T>
svm_summary<T>::svm_summary(std::shared_ptr<T> model, const svm_instance & instance, NodeID tp, NodeID tn, NodeID fp, NodeID fn, std::vector<NodeID> SV_min, std::vector<NodeID> SV_maj) {
        this->model = model;
        this->TP = tp;
        this->FP = fp;
        this->TN = tn;
        this->FN = fn;
        this->Sens = (double)tp / (tp+fn) ;
        this->Spec = (double)tn / (tn+fp) ;
        this->Gmean = std::sqrt(this->Sens * this->Spec);
        this->Acc = (double)(tp+tn) / (tp+tn+fp+fn) ;
        if (tp+fp == 0)              //prevent nan case
                this->PPV = 0;
        else
                this->PPV = (double)tp / (tp+fp);

        if (tn+fn == 0)              //prevent nan case
                this->NPV = 0;
        else
                this->NPV = (double)tn / (tn+fn);

        this->F1 = 2.0*tp / (2*tp+fp+fn);

        this->C = model->param.C;
        this->gamma = model->param.gamma;
        this->C_log = std::log(this->C) / std::log(2);
        this->gamma_log = std::log(this->gamma) / std::log(2);
	this->SV_min = SV_min;
	this->SV_maj = SV_maj;
}


template<class T>
void svm_summary<T>::print() {
        std::cout << std::setprecision(5)
                  << "log C: " << this->C_log
                  << " log g: " << this->gamma_log
                  << std::setprecision(3)
                  << std::fixed
                  << " AC:" << this->Acc
                  << " SN:" << this->Sens
                  << " SP:" << this->Spec
                  << " PPV:" << this->PPV
                  << " NPV:" << this->NPV
                  << " F1:" << this->F1
                  << " GM:" << this->Gmean
                  << std::setprecision(0)
                  << " SV_min:" << this->SV_min.size()
                  << " SV_maj:" << this->SV_maj.size()
                  << " TP:" << this->TP
                  << " TN:" << this->TN
                  << " FP:" << this->FP
                  << " FN:" << this->FN
                  << std::setprecision(3)
                  << std::endl;

        std::cout.unsetf(std::ios_base::floatfield);
}

template<class T>
void svm_summary<T>::print_short() {
        std::cout << std::setprecision(3)
                  << "  \tACC=" << this->Acc
                  << "\tGmean=" << this->Gmean
                  << "\tSVs=" << this->SV_min.size() + this->SV_maj.size()
                  << " (" << this->SV_min.size() << "," << this->SV_maj.size() <<")"
                  << std::endl;

        std::cout.unsetf(std::ios_base::floatfield);
}

template<class T>
NodeID svm_summary<T>::num_SV_min() {
        return this->SV_min.size();
}

template<class T>
NodeID svm_summary<T>::num_SV_maj() {
        return this->SV_maj.size();
}

template class svm_summary<svm_model>;
