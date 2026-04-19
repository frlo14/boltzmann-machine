#pragma once

#include <vector>
#include <cstddef>

class matrix {
public:
    matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }

    const double operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    std::size_t rows() const {
        return rows_;
    }

    std::size_t cols() const{
        return cols_;
    }

private:
    std::size_t rows_, cols_;
    std::vector<double> data_;
};

matrix dotProduct(const matrix& m1, const matrix& m2);
std::vector<double> dotProduct(const matrix& m, const std::vector<double>& v);
double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2);

matrix transpose(const matrix& m);
matrix scalarProduct(double scalar, const matrix& m);

