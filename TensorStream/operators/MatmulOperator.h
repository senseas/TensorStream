#pragma once
#include <vector>
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

    Tensor* A = getInput()[0]->cuda();
    Tensor* B = getInput()[1]->cuda();
    createOutput({A->shape[0], B->shape[1]});
    Tensor* C = this->cuda();

    Tensorx<double> da(A->value, A->valuex, Objects::shapeSize(A->shape) );
    Tensorx<double> db(B->value, B->valuex, Objects::shapeSize(B->shape));
    Tensorx<double> dc(C->value, C->valuex, Objects::shapeSize(C->shape));
    matmul(dc, da, db, A->shape[0], B->shape[1], A->shape[1]);
    return output;
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
      inx->setGrad(out->getGrad() * iny->getValue());
      iny->setGrad(out->getGrad() * inx->getValue());
    });
  }
};