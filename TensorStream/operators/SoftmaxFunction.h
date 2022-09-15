#include "../lang/Include.h"
#include "../lang/Objects.h"
#include "../operators/Operator.h"

using namespace Objects;

class SoftmaNFunction : public TensorFunction {
 public:
  SoftmaNFunction(Tensor* a) : TensorFunction("Softmax", {a}){};

  Object compute() {
    Tenser<Tensor*>* A = getInput<Tenser<Tensor*>*>(0);
    Tenser<Tensor*>* B = zeroTensors(A->shape);
    farEach(A, B, [A](Tensor** a, Tensor** b) {
      *b = new DivOperator(new ExpOperator(*a), new SumOperator(new ExpxFunction(new Tensor(A))));
    });
    return B;
  }
};