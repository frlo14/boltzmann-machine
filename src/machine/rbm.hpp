#pragma once

#include <cstddef>
#include <cmath>
#include <vector>
#include "maths/matrix.hpp"
#include "maths/activations.hpp"

class RBM {
public:
    RBM(int visible_neurons, int hidden_neurons);

    std::vector<double> sample(const std::vector<double>& probs);
    std::vector<double> reconstruct(const std::vector<double>& v);
    void contrastiveDivergence(const std::vector<double>& v, double lr, int k = 1);

private:
    matrix weight_;
    std::vector<double> visible_bias_, hidden_bias_;
    int visible_neurons_;
    int hidden_neurons_;

    std::vector<double> hiddenProb(const std::vector<double>& v);
    std::vector<double> visibleProb(const std::vector<double>& h);
    void updateWeights(const matrix& pos_grad, const matrix& neg_grad, const std::vector<double>& v, const std::vector<double>& visible_sample, const std::vector<double>& hidden_prob_pos, const std::vector<double>& hidden_prob_neg, double lr);
};