#include <math.h>
#include "../lang/Include.h"
#include "../core/TensorConst.h"
#include "../operators/Operator.h"

class SquareFunction : public TensorFunction {
public:
  SquareFunction(Tensor* a, Tensor* b) : TensorFunction("Square", {a, b}) {};

  Object compute() {
    Tensor* a = getInput<Tensor*>(0);
    Tensor* b = getInput<Tensor*>(1);
    return new MulOperator(new TensorConst(0.5), new PowOperator(new MinusOperator(a, b), new TensorConst(2)));
  }
};

class SquarexFunction : public TensorFunction {
public:
  SquarexFunction(Tensor* a, Tensor* b) : TensorFunction("Squarex", {a, b}) {};

  Object compute() {
    Tenser<Tensor*>* A = getInput<Tenser<Tensor*>*>(0);
    Tenser<Tensor*>* B = getInput<Tenser<Tensor*>*>(1);
    Tensor* C = new TensorConst(0);
    forEach(A, B, [&C](Tensor* a, Tensor* b) {
      C = new AddOperator(C, new SquareFunction(a, b));
    });
    return C;
  }
};