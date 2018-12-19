#include "results.h"
#include <iostream>
#include <iomanip>

results::results() {
        this->cur_iteration = -1;
}

void results::next() {
        this->cur_iteration += 1;
}

void results::setFloat(const std::string &tag, float result) {
        if (this->floats.count(tag) <= 0) {
                this->floats[tag] = std::vector<float>(this->cur_iteration + 1, 0);
                this->tag_order.push_back(tag);
        }
        this->floats[tag].resize(this->cur_iteration + 1);
        this->floats[tag][this->cur_iteration] = result;
}

void results::setString(const std::string & tag, const std::string & result) {
        if (this->strings.count(tag) <= 0) {
                this->strings[tag] = std::vector<std::string>(this->cur_iteration + 1);
                this->tag_order.push_back(tag);
        }
        this->strings[tag].resize(this->cur_iteration + 1);
        this->strings[tag][this->cur_iteration] = result;
}

void results::print() {
        std::cout << std::setprecision(3) << std::fixed;
        for (const auto& tag : this->tag_order) {
                if (this->floats.find(tag) != this->floats.end()) {
                        std::vector<float> & ress = this->floats[tag];
                        float average = 0;
                        for (const float res : ress) {
                                average += res;
                        }

                        average /= this->cur_iteration + 1;

                        std::cout << tag << "\t" << average << std::endl;
                }
                else {
                        std::vector<std::string> & strs = this->strings[tag];
                        std::cout << "[" << tag << "]" << std::endl;
                        for (size_t i = 0; i < strs.size(); i++) {
                                std::cout << "fold " << i << ": "
                                          << strs[i] << std::endl;
                        }
                }
        }
}
