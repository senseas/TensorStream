#include <math.h>
#include "../lang/Include.h"
#include "../core/TensorConst.h"
#include "../operators/Operator.h"

class SoftmaxCrossFunction : public TensorFunction {
 public:
  SoftmaxCrossFunction(Tensor* a, Tensor* b) : TensorFunction("SoftmaxCross", {a, b}){};

  Object compute() {
    Tensor *a = getInput<Tensor*>(0), *b = getInput<Tensor*>(1);
    return new MinusxOperator(new MulOperator(a, new LogOperator(b)));
  }
  
};

class SoftmaxCrossxFunction : public TensorFunction {
 public:
  SoftmaxCrossxFunction(Tensor* a, Tensor* b) : TensorFunction("SoftmaxCross", {a, b}){};

  Object compute() {
    Tenser<Tensor*>* A = getInput<Tenser<Tensor*>*>(0);
    Tenser<Tensor*>* B = getInput<Tenser<Tensor*>*>(1);
    Tensor* C = new TensorConst(0);
    forEach(A, B, [&C](Tensor* a, Tensor* b) {
      C = new AddOperator(C, new SoftmaxCrossFunction(a, b));
    });
    return C;
  }

};