#pragma once
#include <vector>
#include "../lang/Include.h"

using namespace ForEach;
using namespace Objects;

class MatmulOperator : public TensorOperator {
 public:
  MatmulOperator(Tensor* a, Tensor* b) : TensorOperator("Matmul", {a, b}){};

  Object compute() {
    Tenser<None*>* A = getInput<Tenser<None*>*>(0);
    Tenser<None*>* B = getInput<Tenser<None*>*>(1);
    Tenser<None*>* C = zeroNones({A->shape[0], B->shape[1]});

    forEach(A->shape[0], B->shape[1], A->shape[1],
    [A, B, C](int i, int l, int j) {
      None* inx = A->get<None*>(i, j);
      None* iny = B->get<None*>(j, l);
      None* out = C->get<None*>(i, l);
      out->value += inx->value * iny->value;
    });
    return C;
  }

  void gradient() {
    Tenser<None*>* A = getInput<Tenser<None*>*>(0);
    Tenser<None*>* B = getInput<Tenser<None*>*>(1);
    Tenser<None*>* C = getOutput<Tenser<None*>*>();

    forEach(A->shape[0], B->shape[1], A->shape[1],
    [A, B, C](int i, int l, int j) {
      None* inx = A->get<None*>(i, j);
      None* iny = B->get<None*>(j, l);
      None* out = C->get<None*>(i, l);
      inx->setGrad(out->grad * iny->value);
      iny->setGrad(out->grad * inx->value);
    });
  }
};