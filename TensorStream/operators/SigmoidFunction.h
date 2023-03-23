#pragma once

#include "../lang/Include.h"
#include "../core/TensorConst.h"
#include "../lang/Objects.h"
#include "../operators/Operator.h"

using namespace Objects;

class SigmoidFunction : public TensorFunction {
public:
  SigmoidFunction(Tensor* a) : TensorFunction("Sigmoid", {a}) {};

  Object compute() {
    Tensor* A = getInput<Tensor*>(0);
    return new DivOperator(new TensorConst(1), new AddOperator(new TensorConst(1), new ExpOperator(new MinusxOperator(A))));
  }

};

class SigmoidxFunction : public TensorFunction {
public:
  SigmoidxFunction(Tensor* a) : TensorFunction("sigmoidx", {a}) {};

  Object compute() {
    shared_ptr<Tenser<Tensor*>> A = getInput<shared_ptr<Tenser<Tensor*>>>(0);
    shared_ptr<Tenser<Tensor*>> B = zeroTensors(A->shape);
    farEach(A, B, [](Tensor** a, Tensor** b) { *b = new SigmoidFunction(*a); });
    return B;
  }

};