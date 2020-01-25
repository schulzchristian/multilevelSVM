#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>
#include <argtable2.h>
#include "svm/svm_flann.h"
#include "tools/timer.h"
#include "definitions.h"

using namespace std;

typedef vector<FeatureVec> MyMat;


// trim from start (in place)
static inline void ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
                                return !std::isspace(ch);
                        }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
                                return !std::isspace(ch);
                        }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
        ltrim(s);
        rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
        ltrim(s);
        return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
        rtrim(s);
        return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
        trim(s);
        return s;
}

enum NORMALIZE_METHOD {
	NONE, LINEAR, GAUSS_NORM
};

struct config {
	int nn_num = 10;
	int label_col = 0;
	string label_min = "1";
	//indicates whether to normalize or scale to [0,1] or neither
	//scaling can preserve null entries
	NORMALIZE_METHOD norm = GAUSS_NORM;
	// read libsvm data instead of csv
	bool libsvm = false;
	bool processed_csv = false;
	string inputfile;
	string outputfile;
};

int parse_args(int argc, char *argv[], config & conf);

void read_csv(const string & filename, MyMat & min_data, vector<int> & maj_data, int label_col = 0, const string & label_min = "-1");

void read_libsvm(const string & filename, MyMat & data, vector<int> & labels, const string & label_min);

void write_csv(const string & filename, const MyMat & data, const vector<int> & labels);

void normalize(MyMat & data);

void scale(MyMat & data);

void split(const MyMat & data, const vector<int> labels, MyMat & min, MyMat & maj);

void write_metis(const vector<vector<Edge>> & edges, const string output);

void write_features(const MyMat & data, const string filename);

int main(int argc, char *argv[]) {
	config conf;

        if (parse_args(argc, argv, conf)) {
                cout << "parse_args error. exiting..." << endl;
                return 1;
        }

        MyMat data;
        vector<int> labels;

        timer t;

        if (conf.libsvm) {
                read_libsvm(conf.inputfile, data, labels, conf.label_min);
                cout << "read libsvm time " << t.elapsed() << endl;
        } else {
                read_csv(conf.inputfile, data, labels, conf.label_col, conf.label_min);
                cout << "read csv time " << t.elapsed() << endl;
        }

        if (conf.processed_csv || conf.label_col != 0 || conf.label_min != "1") {
                write_csv(conf.outputfile + "_processed.csv", data, labels);
        }

        size_t rows = data.size();
        size_t cols = data[0].size();

        cout << "rows: " << rows << " cols: " << cols << endl;

        t.restart();

	switch (conf.norm) {
	case GAUSS_NORM:
                normalize(data);
                cout << "normalization time " << t.elapsed() << endl;
		break;
	case LINEAR:
                scale(data);
                cout << "scale time " << t.elapsed() << endl;
		break;
	case NONE:
		break;
	}

        MyMat min_data;
        MyMat maj_data;

        t.restart();

        split(data, labels, min_data, maj_data);

        cout << "splitting time " << t.elapsed() << endl;

        std::cout << "nodes - min " << min_data.size()
                  << " maj " << maj_data.size() << std::endl;

        write_features(min_data, conf.outputfile + "_min_data");
        write_features(maj_data, conf.outputfile + "_maj_data");

        /*
        t.restart();

        vector<vector<Edge>> min_edges;
        vector<vector<Edge>> maj_edges;
        svm_flann::run_flann(min_data, min_edges, nn_num);

        std::cout << "flann time min " << t.elapsed() << std::endl;
        t.restart();

        svm_flann::run_flann(maj_data, maj_edges, nn_num);

        cout << "flann time maj " << t.elapsed() << endl;
        t.restart();

        write_metis(min_edges, outputfile + "_min_graph");
        write_metis(maj_edges, outputfile + "_maj_graph");

        cout << "flann + export time " << t.elapsed() << endl;
        */

        return 0;
}

int parse_args(int argc, char *argv[], config & conf) {
        // Setup argtable parameters.
        struct arg_end *end                 = arg_end(100);
        struct arg_lit *help                = arg_lit0("h", "help","Print help.");
        struct arg_int *nearest_neighbors   = arg_int0(NULL, "nn", NULL, "Number of nearest neighbors to compute. (default 10)");
        struct arg_int *label_column        = arg_int0(NULL, "label_col", NULL, "column in which the labels are written (starting at 0)");
        struct arg_str *label_minority      = arg_str0(NULL, "minority", NULL, "label/class of the minority class for binary classifications (default \"1\")");
        struct arg_lit *p_csv               = arg_lit0("c", NULL, "export the csv where the categorical attributes where converted to binary");
        struct arg_lit *scale               = arg_lit0(NULL, "scale", "don't normalize just scale to [0,1]");
        struct arg_lit *no_scale            = arg_lit0(NULL, "no_scale", "neither normalize nor scale to [0,1]");
        struct arg_str *file_format         = arg_str0(NULL, "file_format", "[csv|libsvm]", "The format of the input file (default behavior is csv if the file ends with \".csv\")");
        struct arg_str *filename            = arg_strn(NULL, NULL, "FILE", 1, 1, "Path to csv file to process.");
        struct arg_str *filename_output     = arg_str0("o", "output_filename", "OUTPUT", "Specify the name of the output file. \"path[_{label_min}]_{min,maj}_{graph,data}\" will be used as output. default: FILE without extension");

        void* argtable[] = {help, nearest_neighbors, label_column, label_minority, scale, no_scale, p_csv, file_format, filename, filename_output
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
                conf.nn_num = nearest_neighbors->ival[0];
        }

        if (label_column->count > 0) {
                conf.label_col = label_column->ival[0];
        }

        if (scale->count > 0) {
                conf.norm = LINEAR;
        } else if (no_scale->count > 0) {
                conf.norm = NONE;
        } else {
		conf.norm = GAUSS_NORM;
	}

        if (p_csv->count > 0) {
                conf.processed_csv = true;
        }

        if (filename->count > 0) {
		string file(filename->sval[0]);
		conf.inputfile = file;
		if (file.substr(file.size()-4,4) == ".csv") {
			conf.outputfile = file.substr(0, file.size()-4);
		} else {
                        conf.libsvm = true;
			conf.outputfile = conf.inputfile;
		}
	}

	if (filename_output->count > 0) {
                conf.outputfile = filename_output->sval[0];
        }

        if (label_minority->count > 0) {
                conf.label_min = label_minority->sval[0];
		conf.outputfile += "_" + conf.label_min;
        }

	if (file_format->count > 0) {
		string csv_str("csv");
		cout << csv_str << " compare " << file_format->sval[0] << " = " << (csv_str == file_format->sval[0]) << endl;
		if (csv_str == (file_format->sval[0])) {
			conf.libsvm = false;
		} else {
			conf.libsvm = true;
		}
	}

        arg_freetable(argtable,sizeof(argtable)/sizeof(argtable[0]));

        return 0;
}

void read_csv(const string & filename, MyMat & data, vector<int> & labels, int label_col, const string & label_min) {
        enum COL_TYP {
                LABEL,
                NUMERICAL,
                CATEGORICAL
        };

        std::vector<std::vector<std::string>> col_categorical_value;
        std::vector<COL_TYP> col_typs;

        bool first = true;

        ifstream file;
        file.open(filename);
        for (string line; getline(file, line); ) {
                stringstream sep(line);

                // ignore comments
                if (line[0] == '#')
                        continue;


                if (first == true) {
                        // scan over the first entry to get column information
                        size_t col = 0;
                        for (string item; getline(sep, item, ','); ) {
                                try {
                                        stod(item);
                                        col_typs.push_back(NUMERICAL);
                                } catch (...) {
                                        col_typs.push_back(CATEGORICAL);
                                }
                                col++;
                        }
                        col_typs[label_col] = LABEL;
                        col_categorical_value.resize(col);

                        sep.clear();
                        sep.seekg(0);
                        first = false;
                }

                data.push_back(FeatureVec());
                size_t col=0;
                for (string item; getline(sep, item, ','); ) {
                        switch (col_typs[col]) {
			case LABEL:
				{
					if (item == label_min) {
						labels.push_back(1);
					} else {
						labels.push_back(-1);
					}
					// int label = stoi(item);
                                        // labels.push_back(label);
                                        break;
                                }

                        case NUMERICAL:
                                {
                                        trim(item);
                                        if (item == "?" || item == "na") {
                                                data.back().push_back(-1);
                                        }
                                        FeatureData val = stod(item);
                                        data.back().push_back(val);
                                        break;
                                }

                        case CATEGORICAL:
                                {
                                        // convert categorical attributes to integers
                                        int cat = -1;
                                        for (size_t i = 0; i < col_categorical_value[col].size(); i++) {
                                                if (col_categorical_value[col][i] == item) {
                                                        cat = i;
                                                        break;
                                                }
                                        }
                                        if (cat == -1) {
                                                cat = col_categorical_value[col].size();
                                                col_categorical_value[col].push_back(item);
                                        }
                                        data.back().push_back(cat);
                                        break;
                                }
                        }
                        col++;
                }
        }

        for (size_t col = 0; col < col_categorical_value.size(); col++) {
                if (col_typs[col] != CATEGORICAL)
                        continue;
                cout << "Categorical values for col " << col << endl;
                for (const string & v : col_categorical_value[col]) {
                        cout << v << ",";
                }
                cout << endl;
        }


        // convert categorical attributes to binary
        for (size_t row = 0; row < data.size(); row++) {
                for (size_t col = 0; col < col_typs.size()-1; ++col) {
                        if(col_typs[col] != CATEGORICAL)
                                continue;

                        // for (size_t cate_index)
                }
        }
}

void read_libsvm(const string & filename, MyMat & data, vector<int> & labels, const string & label_min) {
        cout << "begin " << filename << endl;

        ifstream file;
        size_t feature_size = 1;
        file.open(filename);

        for (string line; getline(file, line); ) {
                stringstream sep(line);
                string item;

                data.push_back(FeatureVec());
                data.back().reserve(feature_size);

                getline(sep, item, ' ');
                // labels.push_back(stoi(item));
		if (item == label_min) {
			labels.push_back(1);
		} else {
			labels.push_back(-1);
		}

		for (; getline(sep, item, ' '); ) {
                        auto colon_pos = item.find(':');
                        int index = stoi(item.substr(0, colon_pos));
                        FeatureData value = stod(item.substr(colon_pos+1));

                        while (static_cast<int>(data.back().size()) < index - 1) {
                                data.back().push_back(0);
                        }

                        data.back().push_back(value);

                }
                feature_size = std::max(data.back().size(), feature_size);
        }

        for (auto&& row : data) {
                if (row.size() < feature_size)
                        row.resize(feature_size);
        }
}

void write_csv(const string & filename, const MyMat& data, const vector<int> & labels) {
        size_t rows = data.size();
        size_t cols = data[0].size();

        ofstream file;
        file.open(filename);

        timer t;

        for (size_t i = 0; i < rows; ++i) {
                file << labels[i] << ",";
                for (size_t j = 0; j < cols - 1; ++j) {
                        file << data[i][j] << ",";
                }
                file << data[i][cols-1];
                file << endl;
        }

        cout << "finished writing processed csv to " << filename << " in " << t.elapsed() << endl;
}

void normalize(MyMat & data) {
        size_t rows = data.size();
        size_t cols = data[0].size();

        FeatureVec mean(cols, 0);
        FeatureVec stds(cols);
        FeatureVec variance(cols, 0);

        for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                        mean[j] += data[i][j];
                }
        }

        for (size_t j = 0; j < cols; j++) {
                mean[j] /= (FeatureData) rows;
        }

        for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                        variance[j] += pow(data[i][j] - mean[j], 2);
                }
        }

        for (size_t j = 0; j < cols; j++) {
                stds[j] = sqrt(variance[j] / (FeatureData) (rows - 1));
        }

        for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                        data[i][j] = (data[i][j] - mean[j]) / stds[j];
                }
        }
}

void scale(MyMat & data) {
        size_t rows = data.size();
        size_t cols = data[0].size();
        FeatureVec max = FeatureVec(cols,std::numeric_limits<FeatureData>::min());
        FeatureVec min = FeatureVec(cols,std::numeric_limits<FeatureData>::max());

        for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                        if (data[i][j] > max[j]) {
                                max[j] = data[i][j];
                        } else if (data[i][j] < min[j]) {
                                min[j] = data[i][j];
                        }
                }
        }

        for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                        data[i][j] = (data[i][j] - min[j])/(max[j] - min[j]);
                }
        }
}

void split(const MyMat & data, const vector<int> labels, MyMat & min, MyMat & maj) {
        size_t rows = data.size();
        MyMat data_cpy(data);

        for (int i = rows - 1; i >= 0; --i) {
                MyMat * target = nullptr;
                if (labels[i] == 1) {
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

void write_metis(const vector<vector<Edge>> & edges, const string filename) {
        size_t rows = edges.size();
        size_t nodes = rows;
        size_t num_edges = 0; // unidirected edges
        size_t nn = edges[0].size();

        timer t;

        num_edges = rows * nn;

        ofstream file;
        file.open(filename);

        // edges is NOT the number of edges in the unidirectional graph but an upper boundary
        file << nodes << " " << num_edges << " 1" << endl;

        for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < nn; ++j) {
                        int target = edges[i][j].target;
                        float weight = edges[i][j].weight;
                        if (target == i) //exclude self loops
                                continue;
                        file << target + 1 << " " << (int)weight << " ";
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
