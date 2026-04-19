#pragma once
#include "matrix.hpp"

double sigmoid(double x);
matrix sigmoid(const matrix& m);

double relu(double x);
matrix relu(const matrix& m);

// had to rename to avoid conflict with std::tanh
double tan_h(double x);
matrix tan_h(const matrix& m);