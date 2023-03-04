#pragma once

#include <algorithm>
#include <iostream>
#include <random>

std::default_random_engine random_engine;
std::normal_distribution<double> normal(0.05, 0.01);

double normalRandm() {
  return normal(random_engine);
}

class None {
 public:
  double value = 0, grad = 0;
  bool gradre = true, reduce = false;

  None() { value = normalRandm(); }

  None(double value) {
    this->value = value;
    this->gradre = true;
  }

  None(double value, bool isGrad) {
    this->value = value;
    this->gradre = isGrad;
  }

  void setGrad(double grad) { this->grad += grad; }

  void reset() {
    this->reduce = false;
    this->grad = 0;
  }
};
