#include <math.h>
#include "../core/TensorFunction.h"
#include "../lang/Include.h"

class ShapeFunction : public TensorFunction {
 public:
  ShapeFunction(Tensor* a, Tensor* b) : TensorFunction("Pow", {a, b}){};

  Object compute() {
    Tenser<Tensor*>* A = getInput<Tenser<Tensor*>*>(0);
    Tenser<Tensor*>* B = getInput<Tenser<Tensor*>*>(1);
    Tenser<Tensor*>* C = zeroTensors(B->shape);

    B->setData(A->getData());
    return C;
  }
};