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

  ~Tenser() { delete[] this->data; }

  Tenser(std::vector<int> dim) : shape(dim) {
    next = std::accumulate(dim.begin() + 1, dim.end(), 1L, std::multiplies<size_t>());
    sz = this->next * dim[0];
    data = new T[sz];
  }

  Tenser(T* data, std::vector<int> dim) : data(data), shape(dim) {
    next = std::accumulate(dim.begin() + 1, dim.end(), 1L, std::multiplies<size_t>());
    sz = this->next * dim[0];
  }

  Tenser(const Tenser& tenser) {
    data = tenser.getData();
    next = tenser.getNext();
    sz = tenser.size();
  }

  Tenser operator=(const Tenser& tenser) {
    data = tenser.getData();
    next = tenser.getNext();
    sz = tenser.size();
  }

  template <typename M, class... N>
  M get(N... idxn) {
    std::vector<int> idx = {idxn...};
    Tenser<T>* arr = this;
    if (shape.size() == idx.size()) {
      int length = idx.size() - 1;
      for (int i = 0; i < length; i++) {
        arr = arr->get(idx[i]);
      }
      return (M)arr->getValue(idx[length]);
    }
    else {
      int length = idx.size();
      for (int i = 0; i < length; i++) {
        arr = arr->get(idx[i]);
      }
      return (M)arr;
    }
  }

  template <typename M, class... N>
  M getx(N... idxn) {
    std::vector<int> idx = {idxn...};
    Tenser<T>* arr = this;
    if (shape.size() == idx.size()) {
      int length = idx.size() - 1;
      for (int i = 0; i < length; i++) {
        arr = arr->get(idx[i]);
      }
      return (M)arr->getValuex(idx[length]);
    }
    else {
      int length = idx.size();
      for (int i = 0; i < length; i++) {
        arr = arr->get(idx[i]);
      }
      return (M)arr;
    }
  }

  void setData(T* data) { this->data = data; }
  T* getData() { return this->data; }
  size_t getNext() { return next; }
  size_t size() { return sz; }
  void clear() { delete[] this->data; }

private:
  Tenser<T>* get(int idx) {
    assert(idx < static_cast<int>(shape[0]));
    std::vector<int> dim;
    std::copy(shape.begin() + 1, shape.end(), std::back_inserter(dim));
    Tenser<T>* result = new Tenser<T>(data + idx * next, dim);
    return result;
  }

  T* getValuex(int idx) { return &data[idx]; }

  T getValue(int idx) { return data[idx]; }
};