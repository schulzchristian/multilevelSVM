#ifndef SVM_SUMMARY_H
#define SVM_SUMMARY_H

#include <iomanip>

struct svm_summary {
        int TP;
        int TN;
        int FP;
        int FN;
        double Acc;
        double Sens;
        double Spec;
        double Gmean;
        double F1;
        double PPV;
        double NPV;

        int num_SV_p;
        int num_SV_n;

        double C;
        double gamma;

        double C_log;
        double gamma_log;

        bool operator > (const svm_summary new_) const{
                return (this->Gmean > new_.Gmean);
        }

        svm_summary(const svm_model & model, NodeID tp, NodeID tn, NodeID fp, NodeID fn) {
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
                //TODO swap?
                this->num_SV_p = model.nSV[0];
                this->num_SV_n = model.nSV[1];
        }

        void print() {
                std::cout << std::setprecision(3);
                std::cout << "AC:" << this->Acc;
                std::cout << " SN:" << this->Sens;
                std::cout << " SP:" << this->Spec;
                std::cout << " PPV:" << this->PPV;
                std::cout << " NPV:" << this->NPV;
                std::cout << " F1:" << this->F1;
                std::cout << " GM:" << this->Gmean;
                std::cout << std::setprecision(0);
                std::cout << " SV_min:" << this->num_SV_p;
                std::cout << " SV_maj:" << this->num_SV_n;
                std::cout << " TP:" << this->TP;
                std::cout << " TN:" << this->TN;
                std::cout << " FP:" << this->FP;
                std::cout << " FN:" << this->FN << std::endl;
                std::cout << std::setprecision(4);
        }


};

struct summary_cmp_better_gmean_sn
{
        static bool comp (const svm_summary& a, const svm_summary& b)
        {
                float filter_range = 0.02;
                if( (a.Gmean - b.Gmean) > filter_range )         //a has completely better gmean than b
                        return true;
                else{
                        if( (b.Gmean - a.Gmean) > filter_range )     //b has completely better gmean than a
                                return false;
                        else{                                                    //similar gmean
                                return (a.Sens >  b.Sens );    // a has less nSV than b which is better
                        }
                }
        }
};


#endif /* SVM_SUMMARY_H */
