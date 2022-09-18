#pragma once

class None {
private:
  double* value = nullptr, * grad = nullptr;
  bool* reduce = nullptr;

public:
  bool isGrad = true;

  None(double* value, double* grad, bool* reduce, bool isGrad) {
    this->value = value;
    this->grad = grad;
    this->reduce = reduce;
    this->isGrad = isGrad;
  }

  None(double value, bool isGrad) {
    this->value = new double[1] {value};
    this->grad = new double[1] {0};
    this->reduce = new bool[1] {false};
    this->isGrad = isGrad;
  }

  None(double value) {
    this->value = new double[1] {value};
    this->grad = new double[1] {0};
    this->reduce = new bool[1] {false};
    this->isGrad = true;
  }

  ~None() {
    delete[] value;
    delete[] grad;
    delete[] reduce;
  }

  double getValue() {
    return *this->value;
  }

  void setValue(double value) {
    *this->value = value;
  }

  double getGrad() {
    return *this->grad;
  }

  void setGrad(double grad) {
    *this->grad += grad;
  }

  bool getReduce() {
    return *this->reduce;
  }

  void setReduce(bool reduce) {
    *this->reduce = reduce;
  }

  void reset() {
    *this->reduce = false;
    *this->grad = 0;
  }
};