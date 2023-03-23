#pragma once

#include "../lang/ForEach.h"
#include "../core/Tensor.h"
#include "../core/None.h"
#include <memory>
using namespace std;

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

  void run(shared_ptr<Tenser<double>> input, shared_ptr<Tenser<double>> label) {
    setInput(input);
    setLabel(label);
    run();
  }

  void run() {
    tensor->forward();
    this->backward();
    tensor->reducer();
  }

  void forward(shared_ptr <Tenser<double>> input, shared_ptr<Tenser<double>> label) {
    setInput(input);
    setLabel(label);
    tensor->forward();
  }

  void backward() {
    Object& o = tensor->getOutput();
    forEach<None*>(o, [](None* a) { a->setGrad(1); });
    tensor->backward();
  }

  void reduce() { tensor->reducer(); }

  void setInput(shared_ptr<Tenser<double>> o) {
    shared_ptr<Tenser<None*>> a = input->getOutput().get<shared_ptr<Tenser<None*>>>();
    forEach(a, o, [](None* m, double n) { m->setValue(n); });
  }

  void setLabel(shared_ptr<Tenser<double>> o) {
    shared_ptr<Tenser<None*>> a = label->getOutput().get<shared_ptr<Tenser<None*>>>();
    forEach(a, o , [](None* m, double n) { m->setValue(n); });
  }
};