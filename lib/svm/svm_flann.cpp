#include "svm_flann.h"

#include <flann/flann.hpp>
#include <utility>
#include "definitions.h"
#include "tools/timer.h"


void svm_flann::run_flann(const std::vector<FeatureVec> & data, std::vector<std::vector<Edge>> & graph, int num_nn) {
        size_t rows = data.size();
        size_t cols = data[0].size();

        // copy data to a linear vector, because flann needs a pointer
        FeatureVec tmp;
        tmp.reserve(rows*cols);
        for (size_t i = 0; i < rows; ++i) {
                tmp.insert(tmp.end(), data[i].begin(), data[i].end());
        }

        flann::Matrix<FeatureData> mat(tmp.data(), rows, cols);
        flann::Index<flann::L2<FeatureData>> index(mat, flann::KDTreeIndexParams(1));
        index.buildIndex();
        flann::SearchParams params(64);
        params.cores = 1;

        std::vector<std::vector<int>> indices;
        std::vector<std::vector<FeatureData>> distances;

        // (num_nn + 1) because we don't count the vertex it self as neighbor but flann does
        index.knnSearch(mat, indices, distances, num_nn + 1, params);

        graph.clear();
        graph.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
                graph.push_back(std::vector<Edge>(num_nn));

                for (size_t j = 0; j < (size_t) num_nn + 1; ++j) {
                        int target = indices[i][j];
                        float weight = 1 / distances[i][j];

                        if (target == (size_t) i) //exclude self loops
                                continue;

                        Edge edge;
                        edge.target = target;
                        edge.weight = weight;
                        graph.back().push_back(edge);
                }
        }

}
