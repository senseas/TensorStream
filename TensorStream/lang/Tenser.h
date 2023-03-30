#pragma once

#include <assert.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

template <typename T>
class Tenser {
private:
  T* data;
  size_t next;
  size_t sz;
public:
  std::vector<int> shape;

  //~Tenser() { delete[] this->data; }

  Tenser(std::vector<int> dim) : shape(dim) {
    next = std::accumulate(dim.begin() + 1, dim.end(), 1L, std::multiplies<size_t>());
    sz = this->next * dim[0];
    data = new T[sz];
  }

  Tenser(T* data, std::vector<int> dim) : data(data), shape(dim) {
    next = std::accumulate(dim.begin() + 1, dim.end(), 1L, std::multiplies<size_t>());
    sz = this->next * dim[0];
  }

  Tenser(Tenser<T>& tenser) {
    data = tenser.getData();
    next = tenser.getNext();
    sz = tenser.size();
  }

  Tenser operator=(Tenser<T>& tenser) {
    data = tenser.getData();
    next = tenser.getNext();
    sz = tenser.size();
  }

  template <class... N>
  T get(N... idxn) {
    std::vector<int> idx = {idxn...};
    assert(shape.size() == idx.size());
    Tenser<T>* arr = this;
    int length = idx.size() - 1;
    for (int i = 0; i < length; i++) {
      arr = &arr->newTenser(idx[i]);
    }
    return arr->data[idx[length]];
  }

  template <class... N>
  T* getx(N... idxn) {
    std::vector<int> idx = {idxn...};
    assert(shape.size() == idx.size());
    Tenser<T>* arr = this;
    int length = idx.size() - 1;
    for (int i = 0; i < length; i++) {
      arr = &arr->newTenser(idx[i]);
    }
    return &arr->data[idx[length]];
  }

  template <class... N>
  Tenser<T>* getTenser(N... idxn) {
    std::vector<int> idx = {idxn...};
    assert(shape.size() > idx.size());
    Tenser<T>* arr = this;
    int length = idx.size() - 1;
    for (int i = 0; i < length; i++) {
      arr = &arr->newTenser(idx[i]);
    }
    return arr->newTenserx(idx[length]);
  }

  void clear() { 
    for (int i = 0; i < sz; i++) {
      delete data[i];
    }
    delete[] data;
  }

  void setData(T* data) { this->data = data; }
  T* getData() { return this->data; }
  size_t getNext() { return next; }
  size_t size() { return sz; }

private:
  Tenser<T>& newTenser(int idx) {
    assert(idx < static_cast<int>(shape[0]));
    std::vector<int> dim;
    std::copy(shape.begin() + 1, shape.end(), std::back_inserter(dim));
    Tenser<T> result(data + idx * next, dim);
    return result;
  }

  Tenser<T>* newTenserx (int idx) {
    assert(idx < static_cast<int>(shape[0]));
    std::vector<int> dim;
    std::copy(shape.begin() + 1, shape.end(), std::back_inserter(dim));
    Tenser<T>* result = new Tenser<T>(data + idx * next, dim);
    return result;
  }
};