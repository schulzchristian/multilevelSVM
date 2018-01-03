#include <iostream>
#include <iomanip>
#include <cmath>

#include "svm_summary.h"

svm_summary::svm_summary(const svm_model & model, const svm_desc & desc, NodeID tp, NodeID tn, NodeID fp, NodeID fn) {
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

        this->C = model.param.C;
        this->gamma = model.param.gamma;
        this->C_log = std::log(this->C) / std::log(2);
        this->gamma_log = std::log(this->gamma) / std::log(2);

        this->SV_min.reserve(model.nSV[0]);
        for(int i = 0; i < model.nSV[0]; i++) {
                // libSVM uses 1 as first index while we need our NodeID
                NodeID id = model.sv_indices[i] - 1;
                this->SV_min.push_back(id);
                // std::cout << id << ", ";
        }

        this->SV_maj.reserve(model.nSV[1]);
        for(int i = 0; i < model.nSV[1]; i++) {
                int id = model.sv_indices[model.nSV[0] + i] - 1 - desc.num_min;
                this->SV_maj.push_back(id);
                // std::cout << id << ", ";
        }
        // std::cout << "\n";

}

void svm_summary::print() {
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
                  << std::defaultfloat
                  << std::endl;
}

void svm_summary::print_short() {
        std::cout << std::setprecision(2)
                  << std::fixed
                  << "log C=" << this->C_log
                  << "\tlog gamma=" << this->gamma_log
                  << std::setprecision(3)
                  << "  \tACC=" << this->Acc
                  << "\tGmean=" << this->Gmean
                  << "\tSVs=" << this->SV_min.size() + this->SV_maj.size()
                  << " (" << this->SV_min.size() << "," << this->SV_maj.size() <<")"
                  << std::defaultfloat
                  << std::endl;
}

NodeID svm_summary::num_SV_min() {
        return this->SV_min.size();
}

NodeID svm_summary::num_SV_maj() {
        return this->SV_maj.size();
}
