#include <math.h>
#include "../lang/Include.h"
#include "../core/TensorConst.h"
#include "../core/TensorFunction.h"
#include "../operators/Operator.h"

class SigmoidCrossFunction : public TensorFunction {
 public:
  SigmoidCrossFunction(Tensor* a, Tensor* b) : TensorFunction("SigmoidCross", {a, b}){};

  Object compute() {
    Tensor* a = getInput<Tensor*>(0);
    Tensor* b = getInput<Tensor*>(1);
    return new MinusxOperator(new AddOperator(new MulOperator(a, new LogOperator(b)),new MulOperator(new MinusOperator(new TensorConst(1), a),new LogOperator(new MinusOperator(new TensorConst(1), b)))));
  }
};

class SigmoidCrossxFunction : public TensorFunction {
 public:
  SigmoidCrossxFunction(Tensor* a, Tensor* b) : TensorFunction("SigmoidCrossx", {a, b}){};

  Object compute() {
    Tenser<Tensor*>* A = getInput<Tenser<Tensor*>*>(0);
    Tenser<Tensor*>* B = getInput<Tenser<Tensor*>*>(1);
    Tensor* C = new TensorConst(0);
    forEach(A, B, [&C](Tensor* a, Tensor* b) {
      C = new AddOperator(C, new SigmoidCrossFunction(a, b));
    });
    return C;
  }
};