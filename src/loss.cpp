#include <vector>
#include <cstddef>
#include <cmath>
#include "loss.hpp"

double binaryCrossEntropy(const std::vector<double>& prediction, const std::vector<double>& actual) {
    double loss = 0.0;
    for (std::size_t i = 0; i < prediction.size(); i++) {
        loss += actual[i] * std::log(prediction[i] + 1e-8) + (1 - actual[i]) * std::log(1 - prediction[i] + 1e-8);
    }
    return -loss / prediction.size();
}
