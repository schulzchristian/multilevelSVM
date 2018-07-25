#include <iostream>
#include <cmath>

#include "param_search.h"

constexpr int param_search::UDTable[31][30][2];

std::vector<svm_param> param_search::around(float c_center, float c_radius, float c_step,
                                            float g_center, float g_radius, float g_step) {
        float c_min = c_center - c_radius;
        float c_max = c_center + c_radius;
        float g_min = g_center - g_radius;
        float g_max = g_center + g_radius;
        return grid(c_min, c_max, c_step, g_min, g_max, g_step);
}

std::vector<svm_param> param_search::grid(float c_from, float c_to, float c_step,
                                          float g_from, float g_to, float g_step) {
        std::vector<std::pair<float,float> > seq;

        std::vector<float> seq_c = range(c_from, c_to, c_step);
        std::vector<float> seq_g = range(g_from, g_to, g_step);

        for (auto&& c : seq_c) {
                for (auto&& g : seq_g) {
                        seq.push_back(std::make_pair(c,g));
                }
        }

        return seq;
}

std::vector<std::pair<float,float>> param_search::ud(float c_from, float c_to, float g_from, float g_to,
                                                     bool step1, bool inherit, float param_c, float param_g) {
        std::vector<std::pair<float,float> > seq;

        //TODO make this configurable
        int pattern = step1 ? 9 : 5;
        int stage = step1 ? 1 : 2;

        double lg_base = 2;

        double p_center_c, p_center_g;

        if (inherit) {
                p_center_c = param_c;
                p_center_g = param_g;
        }
        else {
                p_center_c = (c_from + c_to) / lg_base ;
                p_center_g = (g_from + g_to) / lg_base ;
        }

        double c_len = (c_to - c_from) / pow(lg_base, stage - 1);
        double g_len = (g_to - g_from) / pow(lg_base, stage - 1);

        double cen_c = p_center_c - (c_len / pow(lg_base, stage));
        double cen_g = p_center_g - (g_len / pow(lg_base, stage));

        double c_max = pow(lg_base, c_to);
        double c_min = pow(lg_base, c_from);
        double g_max = pow(lg_base, g_to);
        double g_min = pow(lg_base, g_from);

        for(int i=0; i < pattern;i++){
                double c = (((param_search::UDTable[pattern][i][0] - 1) * c_len) / (pattern - 1))
                        + cen_c;
                double g = (((param_search::UDTable[pattern][i][1] - 1) * g_len) / (pattern - 1))
                        + cen_g;

                // scale params to range
                c = pow(lg_base, c);
                g = pow(lg_base, g);

                if (c > c_max ){
                        c = c_max + (rand() % 500) * (rand()%2? 1 :-1);
                }
                if(c < c_min ){
                        c = c_min - (rand() % 100) * (rand()%2? 1 :-1);
                        while(c < 0.001){
                                c +=  (rand() % 500);
                        }
                }

                if (g > g_max ){
                        g = g_max + (rand() % 100 * 0.001) * (rand()%2? 1 :-1);
                }
                if(g < g_min ){
                        g = g_min - (rand() % 100 * 0.00001) * (rand()%2? 1 :-1);
                        while(g < 0.00001){
                                g +=  (rand() % 100 * 0.00001);
                        }
                }

                c = log(c) / log(lg_base);
                g = log(g) / log(lg_base);
                // end scale */

                seq.push_back(std::make_pair(c,g));
        }

        return seq;
}
