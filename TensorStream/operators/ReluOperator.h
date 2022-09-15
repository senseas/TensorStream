#pragma once
#include "../lang/Include.h"
#include "../lang/Objects.h"

using namespace Objects;

class ReluOperator : public TensorOperator {
 public:
  ReluOperator(Tensor* a) : TensorOperator("Relu", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(valx > 0 ? valx : 0.1 * valx);
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* out = getOutput<None*>();
    double grad = out->grad;
    inx->setGrad(inx->value > 0 ? grad : 0.1 * grad);
  }
};

class ReluxOperator : public TensorOperator {
 public:
  ReluxOperator(Tensor* a) : TensorOperator("Relux", {a}){};

  Object compute() {
    Tenser<None*>* A = getInput<Tenser<None*>*>(0);
    Tenser<None*>* B = zeroNones(A->shape);
    forEach(A, B, [](None* a, None* b) {
      double value = a->value;
      b->value = value > 0 ? value : 0.1 * value;
    });
    return B;
  }

  void gradient() {
    Tenser<None*>* A = getInput<Tenser<None*>*>(0);
    Tenser<None*>* B = getOutput<Tenser<None*>*>();
    forEach(A, B, [](None* a, None* b) {
      double grad = b->grad;
      a->setGrad(a->value > 0 ? grad : 0.1 * grad);
    });
  }
};