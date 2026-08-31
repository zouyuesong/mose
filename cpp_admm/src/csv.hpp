// csv.hpp - CSV loading, port of golang_mosek/src/opt/types.go loaders.
// Columns are matched by header name (order-independent); unknown columns
// are ignored, mirroring gocsv behavior.
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct InputItem {
    std::string sym;
    double alpha = 0, position = 0, lambda = 0, tlambda = 0, cost = 0;
    double minTrade = 0, maxTrade = 0, minPosition = 0, maxPosition = 0;
};

struct ConstraintItem {
    std::string name;
    double lb = 0, ub = 0;
    std::string type;
};

struct WeightItem {
    std::string name, sym;
    double weight = 0;
};

// Reads a CSV file into a list of row maps: header name -> raw field.
// Returns false with err set on IO failure.
bool readCsvRows(const std::string &path,
                 std::vector<std::unordered_map<std::string, std::string>> &rows,
                 std::string &err);

std::vector<InputItem> loadInputs(const std::string &path, std::string &err);
std::vector<ConstraintItem> loadConstraints(const std::string &path, std::string &err);
std::vector<WeightItem> loadWeights(const std::string &path, std::string &err);
