#include "rbm.hpp"
#include "maths/matrix.hpp"
#include "maths/activations.hpp"
#include <cstddef>
#include <cmath>
#include <random>
#include <iostream>

static double randomUniform(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

RBM::RBM(int visible_neurons, int hidden_neurons)
    : visible_neurons_(visible_neurons), hidden_neurons_(hidden_neurons),
      weight_(visible_neurons, hidden_neurons),
      visible_bias_(visible_neurons, 0.0),
      hidden_bias_(hidden_neurons, 0.0) {
    // small random weight init
    for (int i = 0; i < visible_neurons_; i++) {
        for (int j = 0; j < hidden_neurons_; j++) {
            weight_(i, j) = randomUniform(-0.01, 0.01);
        }
    }
}

// w^T * v + hidden bias
std::vector<double> RBM::hiddenProb(const std::vector<double>& v) {
    std::vector<double> probs = dotProduct(transpose(weight_), v);
    for (int i = 0; i < hidden_neurons_; i++) {
        probs[i] = sigmoid(probs[i] + hidden_bias_[i]);
    }
    return probs;
}

// wh + visible bias
std::vector<double> RBM::visibleProb(const std::vector<double>& h) {
    std::vector<double> probs = dotProduct(weight_, h);
    for (int i = 0; i < visible_neurons_; i++) {
        probs[i] = sigmoid(probs[i] + visible_bias_[i]);
    }
    return probs;
}

// bernoulli sample
std::vector<double> RBM::sample(const std::vector<double>& probs) {
    std::vector<double> bin(probs.size());
    for (int i = 0; i < (int)probs.size(); i++) {
        bin[i] = (randomUniform(0.0, 1.0) < probs[i]) ? 1.0 : 0.0;
    }
    return bin;
}

std::vector<double> RBM::reconstruct(const std::vector<double>& v) {
    std::vector<double> h = hiddenProb(v);
    return visibleProb(h);
}

void RBM::contrastiveDivergence(const std::vector<double>& v, double lr, int k) {
    // positive phase
    std::vector<double> hidden_prob_pos = hiddenProb(v);
    std::vector<double> hidden_sample = sample(hidden_prob_pos);

    // negative phase 
    std::vector<double> visible_sample = v;
    std::vector<double> hidden_prob_neg;
    for (int i = 0; i < k; i++) {
        std::vector<double> visible_prob = visibleProb(hidden_sample);
        visible_sample = sample(visible_prob);
        hidden_prob_neg = hiddenProb(visible_sample);
        hidden_sample = sample(hidden_prob_neg);
    }

    matrix pos_grad = outerProduct(v, hidden_prob_pos);
    matrix neg_grad = outerProduct(visible_sample, hidden_prob_neg);

    updateWeights(pos_grad, neg_grad, v, visible_sample, hidden_prob_pos, hidden_prob_neg, lr);
}

void RBM::updateWeights(const matrix& pos_grad, const matrix& neg_grad, const std::vector<double>& v, const std::vector<double>& visible_sample, const std::vector<double>& hidden_prob_pos, const std::vector<double>& hidden_prob_neg, double lr) {
    // updates weights
    for (int i = 0; i < visible_neurons_; i++) {
        for (int j = 0; j < hidden_neurons_; j++) {
            weight_(i, j) += lr * (pos_grad(i, j) - neg_grad(i, j));
        }
    }

    // updates visible bias
    for (int i = 0; i < visible_neurons_; i++) {
        visible_bias_[i] += lr * (v[i] - visible_sample[i]);
    } 

    // updates hidden bias
    for (int i = 0; i < hidden_neurons_; i++) {
        hidden_bias_[i] += lr * (hidden_prob_pos[i] - hidden_prob_neg[i]);
    }
}
