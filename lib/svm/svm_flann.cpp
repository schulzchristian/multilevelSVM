#include "svm_flann.h"

#include <flann/flann.hpp>
#include <utility>
#include "definitions.h"
#include "timer.h"


void svm_flann::run_flann(const std::vector<FeatureVec> & data, std::vector<std::vector<Edge>> & graph, int num_nn) {
        size_t rows = data.size();
        size_t cols = data[0].size();

        std::cout << "data points: " << rows << " features: " << cols << std::endl;

        // copy data to a linear vector, because flann needs a pointer
        FeatureVec tmp;
        tmp.reserve(rows*cols);
        for (size_t i = 0; i < rows; ++i) {
                tmp.insert(tmp.end(), data[i].begin(), data[i].end());
        }

        timer t;

        flann::Matrix<FeatureData> mat(tmp.data(), rows, cols);
        flann::Index<flann::L2<float>> index(mat, flann::KDTreeIndexParams(1));
        index.buildIndex();
        flann::SearchParams params(64);
        params.cores = 0;

        std::vector<std::vector<int>> indices;
        std::vector<std::vector<float>> distances;

        // (num_nn + 1) because we don't count the vertex it self as neighbor but flann does
        index.knnSearch(mat, indices, distances, num_nn + 1, params);

        graph.clear();
        graph.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
                std::vector<Edge> edges(num_nn);
                for (size_t j = 0; j < (size_t) num_nn + 1; ++j) {
                        int target = indices[i][j];
                        float weight = 1 / distances[i][j];

                        if (target == (size_t) i) //exclude self loops
                                continue;

                        Edge edge;
                        edge.target = target;
                        edge.weight = weight;
                        edges.push_back(edge);
                }
                graph.push_back(std::move(edges));
        }

        std::cout << "build knn " << t.elapsed() << std::endl;
}
