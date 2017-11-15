#include <iostream>
#include "grid_search.h"

grid_search::grid_search(float c_from, float c_to, float c_step, float g_from, float g_to, float g_step) {
        this->c_from = c_from;
        this->c_to = c_to;
        this->c_step = c_step;
        this->g_from = g_from;
        this->g_to = g_to;
        this->g_step = g_step;
}

grid_search::~grid_search() {
}


grid_search grid_search::around(float c_center, float c_radius, float c_step, float g_center, float g_radius, float g_step) {
        float c_min = c_center - c_radius;
        float c_max = c_center + c_radius;
        float g_min = g_center - g_radius;
        float g_max = g_center + g_radius;
        grid_search gs(c_min, c_max, c_step, g_min, g_max, g_step);
        return gs;
}

std::vector<std::pair<float,float> > grid_search::get_sequence() {
        std::vector<std::pair<float,float> > seq;

        std::vector<float> seq_c = range(this->c_from, this->c_to, this->c_step);
        std::vector<float> seq_g = range(this->g_from, this->g_to, this->g_step);

        for (auto&& c : seq_c) {
                for (auto&& g : seq_g) {
                        seq.push_back(std::make_pair(c,g));
                }
        }

        return seq;
}

template<class T>
std::vector<T> grid_search::range(T from, T to, T step) {
        std::vector<T> seq;
        while (true) {
                if (step > 0 && from > to) break;
                if (step < 0 && from < to) break;
                seq.push_back(from);
                from += step;
        }
        return seq;
}
