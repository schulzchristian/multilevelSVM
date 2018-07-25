#ifndef KFOLD_IMPORT_H
#define KFOLD_IMPORT_H

#include "k_fold.h"

class k_fold_import: public k_fold
{
public:
        k_fold_import(int num_exp, int num_iter, const std::string & basename);
        virtual ~k_fold_import();

protected:
        virtual void next_intern() override;

        std::string basename;
        int num_exp;
};

#endif /* KFOLD_IMPORT_H */
