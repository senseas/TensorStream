#pragma once
#include <vector>
#include <iostream>
#include "../lang/Include.h"
#include "../cuda/Function.h"

using namespace Objects;

class MatmulOperator : public TensorOperator {
public:
  MatmulOperator(Tensor* a, Tensor* b) : TensorOperator("Matmul", {a, b}) {};

  Object compute() {
    /*
    Tenser<None*>* A = getInput<Tenser<None*>*>(0);
    Tenser<None*>* B = getInput<Tenser<None*>*>(1);
    Tenser<None*>* C = createOutput({A->shape[0], B->shape[1]});

    forEach(A->shape[0], B->shape[1], A->shape[1],
    [A, B, C](int i, int l, int j) {
      None* inx = A->get<None*>(i, j);
      None* iny = B->get<None*>(j, l);
      None* out = C->get<None*>(i, l);
      out->setValue(out->getValue() + inx->getValue() * iny->getValue());
    });
    */

    Tensorx<double> DA = getInput(0), DB = getInput(1);
    createOutput({DA.shape(0), DB.shape(1)});
    Tensorx<double> DC(this);
    matmul(DA, DB, DC, DA.shape(0), DB.shape(1), DA.shape(1));

    return output;
  }

  void gradient() {
    Tenser<None*>* A = getInput<Tenser<None*>*>(0);
    Tenser<None*>* B = getInput<Tenser<None*>*>(1);
    Tenser<None*>* C = getOutput<Tenser<None*>*>();

    forEach(A->shape[0], B->shape[1], A->shape[1],
    [A, B, C](int i, int l, int j) {
      None* inx = A->get(i, j);
      None* iny = B->get(j, l);
      None* out = C->get(i, l);
      inx->setGrad(out->getGrad() * iny->getValue());
      iny->setGrad(out->getGrad() * inx->getValue());
    });
  }
};