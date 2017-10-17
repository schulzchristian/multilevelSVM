#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <flann/flann.hpp>
#include "../lib/tools/timer.h"

using namespace std;
using namespace Eigen;

typedef double MyItem;
typedef vector<vector<MyItem>> MyMat;
typedef vector<MyItem> MyRow;

void readCSV(const string filename, MyMat & min_data, vector<int> & maj_data);

MatrixXd toMatrixXd(const MyMat & data);

MyMat normalizeIP(const MyMat & data);

void normalize(MyMat & data);

void normalize(MatrixXd & data);

void split(const MatrixXd & m, const vector<int> label, MatrixXd & min, MatrixXd & maj);

void split(const MyMat & data, const vector<int> label, MyMat & min, MyMat & maj);

void run_flann(const MyMat & data, vector<vector<int>> & indices, MyMat & distances, int num_nn);

void write_metis(const vector<vector<int>> & indices, const MyMat & distances, const string output);

void write_features(const MyMat & data, const string filename);

int main(int argc, char *argv[]) {
  string inputfile = argv[1];
  string outputfile = argv[2];

  MyMat data;
  vector<int> label;

  timer t;

  readCSV(inputfile, data, label);

  cout << "read csv took " << t.elapsed() << endl;

  size_t rows = data.size();
  size_t cols = data[0].size();

  t.restart();

  normalize(data);

  cout << "normalization took " << t.elapsed() << endl;

  MyMat min_data;
  MyMat maj_data;

  t.restart();

  split(data, label, min_data, maj_data);

  cout << "splitting took " << t.elapsed() << endl;

  // cout << "min " << min_data.size() << 'x' << min_data[0].size() << endl;
  // cout << "min " << maj_data.size() << 'x' << maj_data[0].size() << endl;

  vector<vector<int>> min_indices;
  MyMat min_distances;
  vector<vector<int>> maj_indices;
  MyMat maj_distances;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      // cout << " " << min_data[i][j];
    }
    // cout << endl;
  }

  t.restart();

  run_flann(min_data, min_indices, min_distances, 10);
  run_flann(maj_data, maj_indices, maj_distances, 10);

  cout << "flann time " << t.elapsed() << endl;

  t.restart();

  write_metis(min_indices, min_distances, outputfile + "_min_graph");
  write_metis(maj_indices, maj_distances, outputfile + "_maj_graph");
  write_features(min_data, outputfile + "_min_data");
  write_features(maj_data, outputfile + "_maj_data");

  cout << "export time " << t.elapsed() << endl;

  return 0;
}

void readCSV(const string filename, MyMat & data, vector<int> & label) {
  ifstream file;
  file.open(filename);

  for (string line; getline(file, line); ) {
    stringstream sep(line);

    string main_class;
    getline(sep, main_class, ',');
    int main_c = stoi(main_class);
    label.push_back(main_c);

    data.push_back(MyRow());

    for (string item; getline(sep, item, ','); ) {
      MyItem val = stod(item);
      data.back().push_back(val);
    }
  }
}

MatrixXd toMatrixXd(const MyMat & data) {
  size_t rows = data.size();
  size_t cols = data[0].size();

  MatrixXd result(rows,cols);
  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
      result(i,j) = data[i][j];

  return result;
}

MyMat normalizeIP(const MyMat & data) {
  size_t rows = data.size();
  size_t cols = data[0].size();
  MyMat data_cpy(data);

  MyRow mean(cols, 0);
  MyRow stds(cols, 0);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mean[j] += data[i][j];
    }
  }

  for (size_t j = 0; j < cols; j++) {
    mean[j] /= (MyItem) rows;
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      stds[j] += pow(data[i][j] - mean[j], 2);
    }
  }

  for (size_t j = 0; j < cols; j++) {
    stds[j] = sqrt(stds[j] / (MyItem) (rows - 1));
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      data_cpy[i][j] = (data[i][j] - mean[j]) / stds[j];
    }
  }

  return data_cpy;
}

void normalize(MyMat & data) {
  size_t rows = data.size();
  size_t cols = data[0].size();

  MyRow mean(cols, 0);
  MyRow stds(cols);
  MyRow variance(cols, 0);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mean[j] += data[i][j];
    }
  }

  for (size_t j = 0; j < cols; j++) {
    mean[j] /= (MyItem) rows;
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      variance[j] += pow(data[i][j] - mean[j], 2);
    }
  }

  for (size_t j = 0; j < cols; j++) {
    stds[j] = sqrt(variance[j] / (MyItem) (rows - 1));
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      data[i][j] = (data[i][j] - mean[j]) / stds[j];
    }
  }
}

void split(const MyMat & data, const vector<int> label, MyMat & min, MyMat & maj) {
  size_t rows = data.size();
  size_t cols = data[0].size();
  MyMat data_cpy(data);

  for (int i = rows - 1; i >= 0; --i) {
    MyMat * target = nullptr;
    if (label[i] == 1) {
      target = &min;
    } else {
      target = &maj;
    }
    target->push_back(data_cpy.back());
    data_cpy.pop_back();
  }

  reverse(min.begin(), min.end());
  reverse(maj.begin(), maj.end());
}

void normalize(MatrixXd & data) {
  // calc mean
  VectorXd mean = data.colwise().sum() / (double) data.rows();

  // (data.array().colwise() - mean).pow(2).sum();

  // calc standard derivation
  VectorXd stds(data.cols());
  for (Index i = 0; i < data.cols(); ++i) {
    ArrayXd tmp = data.col(i).array() - mean(i);
    double variance = tmp.pow(2).sum();
    stds(i) = sqrt(variance / (double) (data.rows() - 1));
  }

  // zscore
  for (Index i = 0; i < data.cols(); ++i) {
    for (Index j = 0; j < data.rows(); ++j) {
      data(j,i) = (data(j,i) - mean(i)) / stds(i);
    }
  }
}

void split(const MatrixXd & m, const vector<int> label, MatrixXd & min, MatrixXd & maj) {

}

void run_flann(const MyMat & data, vector<vector<int>> & indices, MyMat & distances, int num_nn) {
  size_t rows = data.size();
  size_t cols = data[0].size();

  cout << "data points: " << rows << " features: " << cols << endl;

  timer t;
  vector<MyItem> tmp;
  for (size_t i = 0; i < rows; ++i) {
    tmp.insert(tmp.end(), data[i].begin(), data[i].end());
  }

  cout << "build tmp " << t.elapsed() << endl;
  t.restart();

  flann::Matrix<MyItem> mat(tmp.data(), rows, cols);
  flann::Index<flann::L2<MyItem>> index(mat, flann::KDTreeIndexParams(1));
  index.buildIndex();
  flann::SearchParams params(64);
  params.cores = 0;
  index.knnSearch(mat, indices, distances, num_nn, params);

  cout << "build knn " << t.elapsed() << endl;
}

void write_metis(const vector<vector<int>> & indices, const MyMat & distances, const string filename) {
  size_t rows = indices.size();
  size_t nodes = rows;
  size_t edges = 0; // unidirected edges
  size_t nn = indices[0].size();

  timer t;

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < nn; ++j) {
      if (indices[i][j] > i) {
        edges++;
      }
    }
  }

  ofstream file;
  file.open(filename);

  file << nodes << " " << edges << " " << " 1" << endl;

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < nn; ++j) {
      int target = indices[i][j];
      MyItem weight = distances[i][j];
      if (target == i)
        continue;
      file << target + 1 << " " << weight << " ";
    }
    file << endl;
  }

  file.close();

  cout << "finished writing graph to " << filename << " in " << t.elapsed() << endl;
}

void write_features(const MyMat & data, const string filename) {
  size_t rows = data.size();
  size_t cols = data[0].size();

  ofstream file;
  file.open(filename);

  timer t;

  file << rows << " " << cols << endl;

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      file << data[i][j] << " ";
    }
    file << endl;
  }

  cout << "finished writing features to " << filename << " in " << t.elapsed() << endl;
}
