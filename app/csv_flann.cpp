#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <flann/flann.hpp>
#include <argtable2.h>
#include "timer.h"

using namespace std;

typedef double MyItem;
typedef vector<vector<MyItem>> MyMat;
typedef vector<MyItem> MyRow;

int parse_args(int argc, char *argv[], int & nn_num, int & label_col, string & inputfile, string & outputfile);

void readCSV(const string filename, MyMat & min_data, vector<int> & maj_data, int label_col = 0);

MyMat normalizeIP(const MyMat & data);

void normalize(MyMat & data);

void split(const MyMat & data, const vector<int> label, MyMat & min, MyMat & maj);

void run_flann(const MyMat & data, vector<vector<int>> & indices, MyMat & distances, int num_nn);

void write_metis(const vector<vector<int>> & indices, const MyMat & distances, const string output);

void write_features(const MyMat & data, const string filename);

int main(int argc, char *argv[]) {
        int nn_num = 10;
        int label_col = 0;
        string inputfile;
        string outputfile;

        if (parse_args(argc, argv, nn_num, label_col, inputfile, outputfile))
                return 1;

        MyMat data;
        vector<int> label;

        timer t;

        readCSV(inputfile, data, label, label_col);

        cout << "read csv took " << t.elapsed() << endl;

        size_t rows = data.size();
        size_t cols = data[0].size();

        cout << "rows: " << rows << " cols: " << cols << endl;

        t.restart();

        // normalize(data);

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

        t.restart();

        run_flann(min_data, min_indices, min_distances, nn_num);
        run_flann(maj_data, maj_indices, maj_distances, nn_num);

        cout << "flann time " << t.elapsed() << endl;

        t.restart();

        write_metis(min_indices, min_distances, outputfile + "_min_graph");
        write_metis(maj_indices, maj_distances, outputfile + "_maj_graph");
        write_features(min_data, outputfile + "_min_data");
        write_features(maj_data, outputfile + "_maj_data");

        cout << "export time " << t.elapsed() << endl;

        return 0;
}

int parse_args(int argc, char *argv[], int & nn_num, int & label_col, string & inputfile, string & outputfile) {
        // Setup argtable parameters.
        struct arg_end *end                 = arg_end(100);
        struct arg_lit *help                = arg_lit0("h", "help","Print help.");
        struct arg_int *nearest_neighbors   = arg_int0(NULL, "nn", NULL, "Number of nearest neighbors to compute.");
        struct arg_int *label_column        = arg_int0(NULL, "label_col", NULL, "column in which the labels are written (starting at 0)");
        struct arg_str *filename            = arg_strn(NULL, NULL, "FILE", 1, 1, "Path to csv file to process.");
        struct arg_str *filename_output     = arg_str0("o", "output_filename", "OUTPUT", "Specify the name of the output file. \"path_{min,maj}_{graph,data}\" will be used as output. default: FILE without extension");

        void* argtable[] = {help, nearest_neighbors, label_column, filename, filename_output
                            ,end};

        // Parse arguments.
        int nerrors = arg_parse(argc, argv, argtable);

        const char *progname = argv[0];

        // help or error
        if (nerrors > 0 || help->count > 0) {
                printf("Usage: %s", progname);
                arg_print_syntax(stdout, argtable, "\n");
                arg_print_glossary(stdout, argtable,"  %-40s %s\n");
                arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
                return 1;
        }

        if (nearest_neighbors->count > 0) {
                // plus one because we don't count the vertex it self as neighbor but flann does
                nn_num = nearest_neighbors->ival[0] + 1;
        }

        if (label_column->count > 0) {
                label_col= label_column->ival[0];
        }

        if (filename->count > 0) {
                inputfile = filename->sval[0];
                outputfile = inputfile.substr(0,inputfile.find_last_of('.'));
        }

        if (filename_output->count > 0) {
                outputfile = filename_output->sval[0];
        }

        arg_freetable(argtable,sizeof(argtable)/sizeof(argtable[0]));

        return 0;
}

void readCSV(const string filename, MyMat & data, vector<int> & label, int label_col) {
        ifstream file;
        file.open(filename);

        for (string line; getline(file, line); ) {
                stringstream sep(line);

                // ignore comments
                if (line[0] == '#')
                        continue;

                data.push_back(MyRow());

                int col=0;
                for (string item; getline(sep, item, ','); ) {
                        if (col == label_col) {
                                int val = stoi(item);
                                label.push_back(val);
                        } else{
                                MyItem val = stod(item);
                                data.back().push_back(val);
                        }
                        col++;
                }
        }
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

        edges = rows * (nn - 1);

        ofstream file;
        file.open(filename);

        // edges is NOT the number of edges in the unidirectional graph but an upper boundary
        file << nodes << " " << edges << " 1" << endl;

        for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < nn; ++j) {
                        int target = indices[i][j];
                        MyItem weight = distances[i][j];
                        if (target == i) //exclude self loops
                                continue;
                        file << target + 1 << " " << (int)(1 / weight) << " ";
                        // file << target + 1 << " " << weight << " ";
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
