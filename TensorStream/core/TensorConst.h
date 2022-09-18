#pragma once
#include "../core/Tensor.h"

class TensorConst : public Tensor {
public:
  TensorConst(double value) : Tensor(value, false) {};

  TensorConst(vector<int> shape, double value) : Tensor(shape, value, false) {};
};