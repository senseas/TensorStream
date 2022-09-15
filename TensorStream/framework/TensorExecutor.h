#pragma once

#include "../lang/ForEach.h"
#include "../core/Tensor.h"

using namespace ForEach;

class TensorExecutor {
private:
  Tensor* tensor;
  Tensor* input, * label;

public:
  TensorExecutor(Tensor* tensor, Tensor* input, Tensor* label) {
	this->tensor = tensor;
	this->input = input;
	this->label = label;
  }

  void run(Tenser<double>* input, Tenser<double>* label) {
	setInput(input);
	setLabel(label);
	run();
  }

  void run() {
	tensor->forward();
	this->backward();
	tensor->reduce();
  }

  void forward(Tenser<double>* input, Tenser<double>* label) {
	setInput(input);
	setLabel(label);
	tensor->forward();
  }

  void backward() {
	Object& o = tensor->getOutput();
	forEach<None*>(o, [](None* a) { a->setGrad(1); });
	tensor->backward();
  }

  void reduce() { tensor->reduce(); }

  void setInput(Tenser<double>* o) {
	Tenser<None*>* a = input->getOutput().get<Tenser<None*>*>();
	farEach(a, o, [](None** m, double* n) { (*m)->value = *n; });
  }

  void setLabel(Tenser<double>* o) {
	Tenser<None*>* a = label->getOutput().get<Tenser<None*>*>();
	farEach(a, o, [](None** m, double* n) { (*m)->value = *n; });
  }
};