#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "../lang/Object.h"
#include "../lang/Tenser.h"
#include "../core/None.h"

using namespace std;
class Tensor;

namespace Objects {
  Tenser<None*>* random(vector<int> shape);
  Tenser<None*>* fill(vector<int> shape, double value, bool isGrad);
  bool isNone(Tensor* tensor);
  vector<int> shapes(Tenser<Object*>* arr);
  bool isFunction(Tensor* tensor);
};  // namespace Objects

class Tensor {
 public:
  string name = "Tensor::";
  vector<Tensor*> input;
  Object function;
  Object output;

 public:
  Tensor(string name, vector<Tensor*> input) {
    this->name = this->name + name;
    this->input = input;
  }

  Tensor(double value) {
    this->name = "None";
    this->output = new None(value);
  }

  Tensor(double value, bool isGrad) {
    this->name = "None";
    this->output = new None(value, isGrad);
  }

  Tensor(std::initializer_list<int> shape) {
    this->name = "None";
    this->output = Objects::random(shape);
  }

  Tensor(string name, vector<int> shape) {
    this->name = "None::" + name;
    this->output = Objects::random(shape);
  }

  Tensor(vector<int> shape, double value, bool isGrad) {
    this->name = "None";
    this->output = Objects::fill(shape, value, isGrad);
  }

  Tensor(None* input) {
    this->name = "None";
    this->output = input;
  }

  Tensor(void* function) {
    this->name = "Function";
    this->function = function;
  }

  vector<Tensor*> getInput() {
    return input;
  }

  virtual Object compute() { return nullptr; }
  virtual void gradient(){};

  virtual void forward(){};
  virtual void backward(){};
  virtual void reduce(){};

  virtual Object& getOutput() {
    return output;
  }
  virtual Object& getFunction() {
    return function;
  }

  template <typename M>
  M getOutput() {
    return output.get<M>();
  }
};