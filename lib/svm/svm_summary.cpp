#include <iostream>
#include <iomanip>
#include <cmath>

#include "svm_summary.h"

svm_summary::svm_summary(const svm_model & model, NodeID tp, NodeID tn, NodeID fp, NodeID fn) {
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

        //TODO swap?
        this->num_SV_min = model.nSV[0];
        this->num_SV_maj = model.nSV[1];

        this->indices_SV_min.reserve(this->num_SV_min);
        this->indices_SV_maj.reserve(this->num_SV_maj);

        for(int i = 0; i < model.nSV[0]; i++) {
                this->indices_SV_min.push_back(model.sv_indices[i]);
        }
        for(int i = model.nSV[0]; i < model.nSV[1]; i++) {
                this->indices_SV_maj.push_back(model.sv_indices[i]);
        }
}

void svm_summary::print() {
        std::cout << std::setprecision(3)
                  << std::fixed
                  << "AC:" << this->Acc
                  << " SN:" << this->Sens
                  << " SP:" << this->Spec
                  << " PPV:" << this->PPV
                  << " NPV:" << this->NPV
                  << " F1:" << this->F1
                  << " GM:" << this->Gmean
                  << std::setprecision(0)
                  << " SV_min:" << this->num_SV_min
                  << " SV_maj:" << this->num_SV_maj
                  << " TP:" << this->TP
                  << " TN:" << this->TN
                  << " FP:" << this->FP
                  << " FN:" << this->FN
                  << std::setprecision(3)
                  << std::defaultfloat
                  << std::endl;
}

void svm_summary::print_short() {
        std::cout << std::setprecision(3)
                  << "log C=" << this->C_log
                  << " \tlog gamma=" << this->gamma_log
                  << "\tACC=" << this->Acc
                  << "\tGmean=" << this->Gmean
                  << std::endl;
}
