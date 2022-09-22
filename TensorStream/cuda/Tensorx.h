#pragma once
#include <vector>
#include "../cuda/CudaUtil.h"
using namespace std;

template<typename T>
class Tensorx {
private:
  size_t size_;
  vector<int> shape_;
  T* data_ = nullptr, * datax_ = nullptr;

public:
  Tensorx(vector<T> data) {
    this->size_ = data.size();
    this->data_ = data.data();
    this->datax_ = setCudaData(*this);
  }

  Tensorx(size_t size) {
    vector<T> data(size);
    this->size_ = size;
    this->data_ = data.data();
    this->datax_ = setCudaData(*this);
  }

  Tensorx(T* data, size_t size) {
    this->size_ = size;
    this->data_ = data;
    this->datax_ = setCudaData(*this);
  }

  Tensorx(Tensor* a) {
    this->size_ = a->size();
    this->shape_ = a->shape;
    this->data_ = a->value;
    this->datax_ = a->valuex;
    this->datax_ = setCudaData(*this);
    a->valuex = this->datax_;
  }

  size_t size() {
    return size_;
  }

  vector<int>& shape() {
    return shape_;
  }

  int shape(int i) {
    return shape_[i];
  }

  T* data() {
    return data_;
  }

  T* datax() {
    return datax_;
  }

};