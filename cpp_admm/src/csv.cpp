// csv.cpp - implementation, see csv.hpp
#include "csv.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>

static std::vector<std::string> splitLine(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i <= line.size(); i++) {
        if (i == line.size() || line[i] == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += line[i];
        }
    }
    return out;
}

static std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\r' || s[a] == '\t')) a++;
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\r' || s[b-1] == '\t')) b--;
    return s.substr(a, b - a);
}

bool readCsvRows(const std::string &path,
                 std::vector<std::unordered_map<std::string, std::string>> &rows,
                 std::string &err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::string line;
    std::vector<std::string> header;
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields = splitLine(line);
        if (first) {
            for (auto &h : fields) header.push_back(trim(h));
            first = false;
            continue;
        }
        std::unordered_map<std::string, std::string> row;
        for (size_t i = 0; i < fields.size() && i < header.size(); i++) {
            row[header[i]] = trim(fields[i]);
        }
        rows.push_back(std::move(row));
    }
    return true;
}

static double toD(const std::string &s) {
    if (s.empty()) return 0.0;
    return std::strtod(s.c_str(), nullptr);
}

std::vector<InputItem> loadInputs(const std::string &path, std::string &err) {
    std::vector<std::unordered_map<std::string, std::string>> rows;
    std::vector<InputItem> out;
    if (!readCsvRows(path, rows, err)) return out;
    for (auto &r : rows) {
        InputItem it;
        it.sym = r.count("sym") ? r["sym"] : "";
        it.alpha = toD(r["alpha"]);
        it.position = toD(r["position"]);
        it.lambda = toD(r["lambda"]);
        it.tlambda = toD(r["tlambda"]);
        it.cost = toD(r["cost"]);
        it.minTrade = toD(r["minTrade"]);
        it.maxTrade = toD(r["maxTrade"]);
        it.minPosition = toD(r["minPosition"]);
        it.maxPosition = toD(r["maxPosition"]);
        out.push_back(it);
    }
    return out;
}

std::vector<ConstraintItem> loadConstraints(const std::string &path, std::string &err) {
    std::vector<std::unordered_map<std::string, std::string>> rows;
    std::vector<ConstraintItem> out;
    if (!readCsvRows(path, rows, err)) return out;
    for (auto &r : rows) {
        ConstraintItem it;
        it.name = r.count("name") ? r["name"] : "";
        it.lb = toD(r["lb"]);
        it.ub = toD(r["ub"]);
        it.type = r.count("type") ? r["type"] : "";
        out.push_back(it);
    }
    return out;
}

std::vector<WeightItem> loadWeights(const std::string &path, std::string &err) {
    std::vector<std::unordered_map<std::string, std::string>> rows;
    std::vector<WeightItem> out;
    if (!readCsvRows(path, rows, err)) return out;
    for (auto &r : rows) {
        WeightItem it;
        it.name = r.count("name") ? r["name"] : "";
        it.sym = r.count("sym") ? r["sym"] : "";
        it.weight = toD(r["weight"]);
        out.push_back(it);
    }
    return out;
}
