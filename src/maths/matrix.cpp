#include "matrix.hpp"
#include <vector>
#include <stdexcept>

// matrix x matrix 
matrix dotProduct(const matrix& m1, const matrix& m2) {
    if (m1.cols() != m2.rows()) {
        throw std::invalid_argument("dimensions dont match");
    }

    matrix result(m1.rows(), m2.cols());
    for (std::size_t i = 0; i < m1.rows(); i++) {
        for (std::size_t j = 0; j < m2.cols(); j++) {
            result(i, j) = 0.0;
            for (std::size_t k = 0; k < m1.cols(); k++) {
                result(i, j) += m1(i, k) * m2(k, j);
            }
        }
    }

    return result;
}

// matrix x vector
std::vector<double> dotProduct(const matrix& m, const std::vector<double>& v) {
    if (m.cols() != v.size())
        throw std::invalid_argument("dimensions dont match");

    std::vector<double> result(m.rows(), 0.0);

    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t k = 0; k < m.cols(); k++) {
            result[i] += m(i, k) * v[k];
        }
    }

    return result;
}

// vector x vector
double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size())
        throw std::invalid_argument("dimensions dont match");

    double result = 0.0;

    for (std::size_t i = 0; i < v1.size(); i++) {
        result += v1[i] * v2[i];
    }

    return result;
}

matrix transpose(const matrix& m) {
    matrix result(m.cols(), m.rows());
    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            result(j, i) = m(i, j);
        }
    }

    return result;
}

matrix scalarProduct(const double& scalar, const matrix& m) {
    matrix result(m.rows(), m.cols());
    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            result(i, j) = scalar * m(i, j);
        }
    }

    return result;
}

matrix outerProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    matrix result(v1.size(), v2.size());
    for (std::size_t i = 0; i < v1.size(); i++) {
        for (std::size_t j = 0; j < v2.size(); j++) {
            result(i, j) = v1[i] * v2[j];
        }
    }
    
    return result;
}