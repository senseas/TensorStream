#pragma once
#include "../core/TensorFlux.h"

class TensorOperator : public Tensor {
public:
  TensorOperator(string name, vector<Tensor*> input) : Tensor(name, input) {};

  template <typename M>
  M getInput(int i) {
    Tensor* input = getInput()[i];
    if (Objects::isFunction(input)) {
      return TensorFlux::getOutput(input->getFunction()).get<M>();
    }
    return input->getOutput().get<M>();
  }

  vector<Tensor*> getInput() { return input; }

  Tenser<None*>* createOutput(vector<int> shape) {
    if (this->getOutput().isNull()) {
      TensorFlux::createOutput(this, shape);
    }
    return output.get<Tenser<None*>*>();
  }

  None* createOutput() {
    if (this->getOutput().isNull()) {
      TensorFlux::createOutput(this, shape);
    }
    return output.get<None*>();
  }

  void forward() {
    for (Tensor* o : getInput()) {
      TensorFlux::computer(o);
    }
    TensorFlux::compute(this);
  }

  void backward() {
    TensorFlux::gradient(this);
    for (Tensor* o : getInput()) {
      o->backward();
    }
  }

  void reducer() {
    for (Tensor* o : getInput()) {
      TensorFlux::reducer(o);
    }
  }
};