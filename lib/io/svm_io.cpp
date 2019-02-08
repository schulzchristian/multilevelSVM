#include "svm_io.h"

#include <fstream>
#include <sstream>
#include <iostream>

#include "svm/svm_convert.h"

void svm_io::readFeaturesLines(const std::string & filename, std::vector<FeatureVec> & data) {
        std::string line;

        // open file for reading
        std::ifstream in(filename);
        if (!in) {
                std::cerr << "Error opening file " << filename << std::endl;
                exit(1);
        }

        std::getline(in, line);
        std::stringstream s(line);
        int nodes = 0;
        int features = 0;
        s >> nodes;
        s >> features;

        data.reserve(nodes);

        while(std::getline(in, line)) {
                std::stringstream ss(line);

                FeatureVec vec(features);
                for (int i = 0; i < features; i++) {
                        ss >> vec[i];
                }

                data.push_back(vec);
        }
}


void svm_io::readTestSplit(const std::string & filename, std::vector<svm_feature> & min_test_data,
                           std::vector<svm_feature> & maj_test_data) {
        std::string line;

        // open file for reading
        std::ifstream in(filename);
        if (!in) {
                std::cerr << "Error opening file " << filename << std::endl;
                exit(1);
        }

        std::getline(in, line);
        std::stringstream s(line);
        int nodes = 0;
        int features = 0;
        s >> nodes;
        s >> features;
        features -= 1; // the label is also counted as feature

        min_test_data.reserve(nodes/2);
        maj_test_data.reserve(nodes);

        while(std::getline(in, line)) {
                std::stringstream ss(line);

                float label;
                ss >> label;

                FeatureVec vec(features);
                for (int i = 0; i < features; i++) {
                        ss >> vec[i];
                }

                if (label == 1) {
                        min_test_data.push_back(svm_convert::feature_to_node(vec));
                } else {
                        maj_test_data.push_back(svm_convert::feature_to_node(vec));
                }
        }
}
