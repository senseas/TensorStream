#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "../lang/Object.h"
#include "../lang/Tenser.h"

using namespace std;
class None;
class Tensor;

namespace Objects {
  bool isNone(Tensor* tensor);
  vector<int> shapes(Tenser<Object*>* arr);
  bool isFunction(Tensor* tensor);

  template <typename T>
  T* random(vector<int> shape);

  template <typename T>
  T* zeros(vector<int> shape);

  template <typename T>
  T* listof(vector<int> shape, T value);

  None* zeroNone(Tensor* tensor, bool isGrad);

  Tenser<None*>* zeroNones(Tensor* tensor, bool isGrad);
};

using namespace Objects;

class Tensor {
public:
  string name = "";
  vector<Tensor*> input;
  vector<int> shape;
  Object function, funcout, output;
  double* value = nullptr, * grad = nullptr;
  bool* reduce = nullptr;

public:
  Tensor(double value) {
    this->name = "None";
    this->value = new double[1] {value};
    this->grad = new double[1] {0};
    this->reduce = new bool[1] {false};
    this->output = zeroNone(this, true);
  }

  Tensor(double value, bool isGrad) {
    this->name = "None";
    this->value = new double[1] {value};;
    this->grad = new double[1] {0};
    this->reduce = new bool[1] {false};
    this->output = zeroNone(this, isGrad);
  }

  Tensor(std::initializer_list<int> shape) {
    this->name = "None";
    this->shape = shape;
    this->value = random<double>(shape);
    this->grad = zeros<double>(shape);
    this->reduce = listof(shape, false);
    this->output = zeroNones(this, true);
  }

  Tensor(string name, vector<int> shape) {
    this->name = "None::" + name;
    this->shape = shape;
    this->value = random<double>(shape);
    this->grad = zeros<double>(shape);
    this->reduce = listof(shape, false);
    this->output = zeroNones(this, true);
  }

  Tensor(vector<int> shape, double value, bool isGrad) {
    this->name = "None";
    this->shape = shape;
    this->value = listof(shape, value);
    this->grad = zeros<double>(shape);
    this->reduce = listof(shape, false);
    this->output = zeroNones(this, isGrad);
  }

  Tensor(string name, vector<Tensor*> input) {
    this->name = this->name + name;
    this->input = input;
  }

  Tensor(None* input) {
    this->name = "None";
    this->output = input;
  }

  Tensor(void* function) {
    this->name = "Function";
    this->function = function;
  }

  ~Tensor() {
    delete[] value;
    delete[] grad;
    delete[] reduce;
  }

  vector<Tensor*> getInput() {
    return input;
  }

  virtual Object compute() { return nullptr; }
  virtual void gradient() {};

  virtual  void forward() {};
  virtual  void backward() {};
  virtual  void reducer() {};

  virtual Object& getOutput() {
    return output;
  }

  template <typename M>
  M getOutput() {
    return output.get<M>();
  }

  virtual Object& getFunction() {
    return function;
  }
  
};