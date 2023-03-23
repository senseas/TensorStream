#pragma once
#include "../core/TensorFlux.h"

class TensorFunction : public Tensor {
public:
  TensorFunction(string name, vector<Tensor*> input) : Tensor(name, input) {};

  template <typename M>
  M getInput(int i) {
    Tensor* input = getInput()[i];
    if (Objects::isFunction(input)) {
      return (input->getFunction()).get<M>();
    }
    return TensorFlux::getTensor(input->getOutput()).get<M>();
  }

  vector<Tensor*> getInput() { return input; }

  Object& getFunction() {
    if (function.nonNull()) {
      return function;
    }
    return function = compute();
  }

  void forward() {
    for (Tensor* o : getInput()) {
      TensorFlux::computer(o);
    }
    TensorFlux::forward(this);
  }

  void backward() {
    TensorFlux::backward(this);
    for (Tensor* o : getInput()) {
      o->backward();
    }
  }

  void reducer() {
    TensorFlux::reduce(this);
    for (Tensor* o : getInput()) {
      TensorFlux::reducer(o);
    }
  }
};