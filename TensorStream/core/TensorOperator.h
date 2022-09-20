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