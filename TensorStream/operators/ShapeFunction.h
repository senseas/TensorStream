#include <math.h>
#include "../core/TensorFunction.h"
#include "../lang/Include.h"

class ShapeFunction : public TensorFunction {
public:
  ShapeFunction(Tensor* a, Tensor* b) : TensorFunction("Pow", {a, b}) {};

  Object compute() {
    shared_ptr<Tenser<Tensor*>> A = getInput<shared_ptr<Tenser<Tensor*>>>(0);
    shared_ptr<Tenser<Tensor*>> B = getInput<shared_ptr<Tenser<Tensor*>>>(1);
    shared_ptr<Tenser<Tensor*>> C = zeroTensors(B->shape);

    B->setData(A->getData());
    return C;
  }
};