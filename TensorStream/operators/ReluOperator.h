#pragma once
#include "../lang/Include.h"
#include "../lang/Objects.h"

using namespace Objects;

class ReluOperator : public TensorOperator {
public:
  ReluOperator(Tensor* a) : TensorOperator("Relu", {a}) {};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->getValue();
    return new None(valx > 0 ? valx : 0.1 * valx);
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* out = getOutput<None*>();
    double grad = out->getGrad();
    inx->setGrad(inx->getValue() > 0 ? grad : 0.1 * grad);
  }
};

class ReluxOperator : public TensorOperator {
public:
  ReluxOperator(Tensor* a) : TensorOperator("Relux", {a}) {};

  Object compute() {
    shared_ptr<Tenser<None*>> A = getInput<shared_ptr<Tenser<None*>>>(0);
    shared_ptr<Tenser<None*>> B = zeroNones(A->shape);
    forEach(A, B, [](None* a, None* b) {
      double value = a->getValue();
      b->setValue(value > 0 ? value : 0.1 * value);
    });
    return B;
  }

  void gradient() {
    Tenser<None*>* A = getInput<Tenser<None*>*>(0);
    Tenser<None*>* B = getOutput<Tenser<None*>*>();
    forEach(A, B, [](None* a, None* b) {
      double grad = b->getGrad();
      a->setGrad(a->getValue() > 0 ? grad : 0.1 * grad);
    });
  }
};