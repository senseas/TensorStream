#pragma once
#include <math.h>
#include "../lang/Include.h"
#include "../lang/Objects.h"

using namespace Objects;

class MaxpoolOperator : public TensorOperator {
 public:
  vector<int> stride;
  int padding = 0, kernelSize = 0;
  MaxpoolOperator(int kernelSize, vector<int> stride, int padding, Tensor* a) : TensorOperator("Maxpool", {a}) {
    this->stride = stride;
  };

  Object compute() {
    Tenser<None*>* A = setPadding(getInput<Tenser<None*>*>(0), padding);
    int heighs = stride[0], widths = stride[1];
    int height = (A->shape[0] - kernelSize) / heighs + 1;
    int width = (A->shape[1] - kernelSize) / widths + 1;
    Tenser<None*>* B = zeroNones({height, width});
    forEach(height, width, kernelSize, kernelSize, [A, B, heighs, widths](int y, int x, int m, int n) {
			None* inx = A->get<None*>(y * heighs + m, x * widths + n);
			None* out = B->get<None*>(y, x);
			out->value = max(out->value, inx->value);
		});
    return B;
  }

  void gradient() {
    Tenser<None*>* A = setPadding(getInput<Tenser<None*>*>(0), padding);
    Tenser<None*>* B = getOutput<Tenser<None*>*>();
    int heighs = stride[0], widths = stride[1];
    forEach(B->shape[1], B->shape[0], kernelSize, kernelSize, [A, B, heighs, widths](int y, int x, int m, int n) {
			None* inx = A->get<None*>(y * heighs + m, x * widths + n);
			None* out = B->get<None*>(y, x);
			inx->setGrad(inx->value == out->value ? out->grad : 0);
		});
  }
};
