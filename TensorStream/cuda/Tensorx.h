#pragma once
#include <vector>
#include "../cuda/CudaUtil.h"
using namespace std;

template<typename T>
class Tensorx {
private:
  size_t size_;
  vector<int> shape;
  T* data_, * datax_;

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

  Tensorx(T* data, T* datax, size_t size) {
    this->size_ = size;
    this->data_ = data;
    this->datax_ = datax;
  }

  size_t size() {
    return size_;
  }

  T* data() {
    return data_;
  }

  T* datax() {
    return datax_;
  }
};