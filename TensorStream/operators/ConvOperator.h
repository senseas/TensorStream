#pragma once
#include "../lang/Include.h"
#include "../lang/Objects.h"
using namespace Objects;

class ConvOperator : public TensorOperator {
private:
  vector<int> stride;
  int padding = 0;

public:
  ConvOperator(vector<int> stride, int padding, Tensor* a, Tensor* b) : TensorOperator("Conv", {a, b}) {
	this->stride = stride;
	this->padding = padding;
  };

  Object compute() {
	Tenser<None*>* A = getInput<Tenser<None*>*>(0);
	Tenser<None*>* B = setPadding(getInput<Tenser<None*>*>(1), padding);
	int heights = stride[0], widths = stride[1];
	int height = (B->shape[0] - A->shape[0]) / heights + 1;
	int width = (B->shape[1] - A->shape[1]) / widths + 1;
	Tenser<None*>* C = zeroNones({height, width});
	forEach(height, width, A->shape[1], A->shape[0],
	[A, B, C, heights, widths](int h, int w, int m, int n) {
	  None* inx = A->get<None*>(m, n);
	  None* iny = B->get<None*>(h * heights + m, w * widths + n);
	  None* out = C->get<None*>(h, w);
	  out->setValue(out->getValue() + inx->getValue() * iny->getValue());
	});
	return C;
  }

  void gradient() {
	Tenser<None*>* A = getInput<Tenser<None*>*>(0);
	Tenser<None*>* B = setPadding(getInput<Tenser<None*>*>(1), padding);
	Tenser<None*>* C = getOutput<Tenser<None*>*>();
	int heights = stride[0], widths = stride[1];
	forEach(C->shape[1], C->shape[0], A->shape[1], A->shape[0],
	[A, B, C, heights, widths](int h, int w, int m, int n) {
	  None* inx = A->get<None*>(m, n);
	  None* iny = B->get<None*>(h * heights + m, w * widths + n);
	  None* out = C->get<None*>(h, w);
	  inx->setGrad(out->getGrad() * iny->getValue());
	  iny->setGrad(out->getGrad() * inx->getValue());
	});
  }
};