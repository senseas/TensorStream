#pragma once
#include <iostream>
#include <vector>
#include "core/Tensor.h"
#include "framework/TensorExecutor.h"
#include "framework/TensorFlow.h"
#include "lang/Objects.h"
#include "lang/Tenser.h"
#include "cuda/Function.h"
using namespace std;

void NNTest() {
  double* inputd = new double[42] {
    0.1, 0.1,
    0.1, 0.2,
    0.1, 0.3,
    0.2, 0.2,
    0.2, 0.3,
    0.2, 0.5,
    0.3, 0.3,
    0.3, 0.4,
    0.3, 0.7,
    0.4, 0.5,
    0.4, 0.6,
    0.4, 0.8,
    0.5, 0.3,
    0.5, 0.6,
    0.5, 0.9,
    0.8, 0.2,
    0.8, 0.7,
    0.8, 0.9,
    0.9, 0.3,
    0.9, 0.6,
    0.9, 0.9
  };

  double* labeld = new double[21] {
    0.01, 0.02, 0.03,
    0.04, 0.06, 0.10,
    0.09, 0.12, 0.21,
    0.20, 0.24, 0.32,
    0.15, 0.30, 0.45,
    0.16, 0.56, 0.72,
    0.27, 0.45, 0.81
  };

  Tenser<double>* inputSet = new Tenser<double>(inputd, {21, 2, 1});
  Tenser<double>* labelSet = new Tenser<double>(labeld, {21, 1, 1});

  Tensor* input = new Tensor({2, 1});
  Tensor* label = new Tensor({1, 1});

  Tensor* tensor11 = Flow::matmul(new Tensor("weight", {4, 2}), input);
  Tensor* tensor12 = Flow::addx(tensor11, new Tensor("bias", {4, 1}));
  Tensor* tensor13 = Flow::sigmoidx(tensor12);

  Tensor* tensor21 = Flow::matmul(new Tensor("weight", {6, 4}), tensor13);
  Tensor* tensor22 = Flow::addx(tensor21, new Tensor("bias", {6, 1}));
  Tensor* tensor23 = Flow::sigmoidx(tensor22);

  Tensor* tensor31 = Flow::matmul(new Tensor("weight", {1, 6}), tensor23);
  Tensor* tensor32 = Flow::addx(tensor31, new Tensor("bias", {1, 1}));
  Tensor* tensor33 = Flow::sigmoidx(tensor32);
  Tensor* tensor34 = Flow::squarex(label, tensor33);

  TensorExecutor* executor = new TensorExecutor(tensor34, input, label);
  std::cout.precision(15);

  forEach(100000000, [inputSet, labelSet, executor, tensor33, tensor34](int i) {
    int l = rand() % (21);
    Tenser<double>* inSet = inputSet->getx<Tenser<double>*>(l);
    Tenser<double>* labSet = labelSet->getx<Tenser<double>*>(l);
    executor->run(inSet, labSet);
    if (i % 100 == 0) {
      None* loss = tensor34->getOutput<None*>();
      Tenser<None*>* out = tensor33->getOutput<Tenser<None*>*>();
      forEach(inSet, [](double n) { std::cout << n << std::endl; });
      forEach(labSet, [](double n) { std::cout << n << std::endl; });
      forEach(out, [](None* a) { std::cout << a->getValue() << std::endl; });
      std::cout << "-----------------" << std::endl;
    }
    delete inSet;
    delete labSet;
  });
}

int main() {
  creatDevice();
  NNTest();
  cudaReset();
  return 0;
}