#pragma once
#include <vector>
using namespace std;

template<typename T>
class Tensorx {
public:
  vector<int> shape;
  vector<T> data;
  T* datax;

public:
  Tensorx(vector<T>& data) {
    this->shape = {(int)data.size()};
    this->data = data;
    this->datax = setCudaData(data);
  }

  Tensorx(vector<T>& data, vector<int>& shape) {
    this->shape = shape;
    this->data = data;
    this->datax = setCudaData(data);
  }
};