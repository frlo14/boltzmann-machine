#include <cstddef>
#include <cmath>
#include "matrix.hpp"
#include "activations.hpp"

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

matrix sigmoid(const matrix& m) {
    matrix result(m.rows(), m.cols());
    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            result(i, j) = sigmoid(m(i, j));
        }
    }

    return result;
}

double relu(double x) {
    if (x < 0) {
        return 0;
    }
    else {
        return x;
    }
}

matrix relu(const matrix& m) {
    matrix result(m.rows(), m.cols());
    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            result(i, j) = relu(m(i, j));
        }
    }

    return result;
}

double tan_h(double x) {
    return (std::exp(x) - std::exp(-x))/ (std::exp(x) + std::exp(-x));
}

matrix tan_h(const matrix& m) {
    matrix result(m.rows(), m.cols());
    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            result(i, j) = tan_h(m(i, j));
        }
    }

    return result;
}