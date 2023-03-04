#pragma once
#include <vector>
using namespace std;

template <typename T>
class Tensor {
 public:
  vector<int> shape;
  vector<T> data;
  T* datax;

 public:
  Tensor(vector<T>& data) {
    this->shape = {(int)data.size()};
    this->data = data;
    this->datax = setCudaData(data);
  }

  Tensor(vector<T>& data, vector<int>& shape) {
    this->shape = shape;
    this->data = data;
    this->datax = setCudaData(data);
  }
};