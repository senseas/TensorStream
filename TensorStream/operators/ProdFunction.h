#include "../lang/Include.h"
#include "../core/TensorConst.h"
#include "../operators/Operator.h"

class ProdFunction : public TensorFunction {
public:
  ProdFunction(Tensor* a, Tensor* b) : TensorFunction("Sigmoid", {a, b}) {};

  Object compute() {
    Tensor* A = getInput<Tensor*>(0);
    return new DivOperator(new TensorConst(1), new AddOperator(new TensorConst(1), new ExpOperator(new MinusxOperator(A))));
  }

};