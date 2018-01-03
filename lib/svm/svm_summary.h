#ifndef SVM_SUMMARY_H
#define SVM_SUMMARY_H

#include <vector>
#include <svm.h>
#include "definitions.h"
#include "svm_desc.h"

class svm_summary {
public:
        svm_summary(const svm_model & model, const svm_desc & desc, NodeID tp, NodeID tn, NodeID fp, NodeID fn);

        void print();
        void print_short();

        NodeID num_SV_min();
        NodeID num_SV_maj();

        bool operator > (const svm_summary new_) const{
                return (this->Gmean > new_.Gmean);
        }

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

        std::vector<NodeID> SV_min;
        std::vector<NodeID> SV_maj;

        double C;
        double gamma;

        double C_log;
        double gamma_log;
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
                                return (a.Sens >  b.Sens );    // a has a better sensitivity
                        }
                }
        }
};

struct summary_cmp_better_gmean_sv
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
                                // a has less nSV than b which is better
                                return (a.SV_min.size() + a.SV_maj.size() <  b.SV_min.size() + b.SV_maj.size());
                        }
                }
        }
};

#endif /* SVM_SUMMARY_H */
