#ifndef RESULTS_H
#define RESULTS_H

#include <vector>
#include <string>
#include <unordered_map>

class results
{
public:
        results();
        void next();
        void setFloat(const std::string & tag, float result);
        void setString(const std::string & tag, const std::string & result);
        void print();

private:
        int cur_iteration;
        std::unordered_map<std::string, std::vector<float>> floats;
        std::unordered_map<std::string, std::vector<std::string>> strings;
        std::vector<std::string> tag_order;
};


#endif /* RESULTS_H */
