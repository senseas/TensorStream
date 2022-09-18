
#pragma once
#include "../lang/Object.h"
#include "../lang/Tenser.h"

namespace ForEach {

  template <typename Func>
  void forEach(int a, Func func) {
    for (int i = 0; i < a; i++) {
      func(i);
    }
  }

  template <typename Func>
  void forEach(int a, int b, Func func) {
    for (int i = 0; i < a; i++) {
      for (int l = 0; l < b; l++) {
        func(i, l);
      }
    }
  }

  template <class Func>
  void forEach(int a, int b, int c, Func func) {
    for (int i = 0; i < a; i++) {
      for (int l = 0; l < b; l++) {
        for (int j = 0; j < c; j++) {
          func(i, l, j);
        }
      }
    }
  }

  template <class Func>
  void forEach(int a, int b, int c, int d, Func func) {
    for (int i = 0; i < a; i++) {
      for (int l = 0; l < b; l++) {
        for (int j = 0; j < c; j++) {
          for (int k = 0; k < d; k++) {
            func(i, l, j, k);
          }
        }
      }
    }
  }

  int shapeSize(vector<int> shape) {
    int size = accumulate(shape.begin(), shape.end(), 1L, std::multiplies<size_t>());
    return size;
  }

  template <typename M, typename Func>
  void farEachi(Tenser<M>* a, Func func) {
    M* data = a->getData();
    for (int i = 0; i < a->size(); i++) {
      M* n = &data[i];
      func(n, i);
    }
  }

  template <typename M, typename Func>
  void farEach(Tenser<M>* a, Func func) {
    for (int i = 0; i < a->size(); i++) {
      M* n = &a->getData()[i];
      func(n);
    }
  }

  template <typename M, typename Func>
  void forEach(Tenser<M>* a, Func func) {
    for (int i = 0; i < a->size(); i++) {
      M n = a->getData()[i];
      func(n);
    }
  }

  template <typename M, typename Func>
  void  forEach(Object& a, Func func) {
    if (a.type() == typeid(Tenser<M>*)) {
      Tenser<M>* m = a.get<Tenser<M>*>();
      forEach(m, func);
    }
    else {
      M m = a.get<M>();
      func(m);
    }
  }

  template <typename M, typename N, typename Func>
  void forEach(Tenser<M>* a, Tenser<N>* b, Func func) {
    for (int i = 0; i < a->size(); i++) {
      M m = a->getData()[i];
      N n = b->getData()[i];
      func(m, n);
    }
  }

  template <typename M, typename Func>
  void forEach(Object& a, Object& b, Func func) {
    if (a.type() == typeid(Tenser<M>*)) {
      Tenser<M>* m = a.get<Tenser<M>*>();
      Tenser<M>* n = b.get<Tenser<M>*>();
      forEach(m, n, func);
    }
    else {
      M m = a.get<M>();
      M n = b.get<M>();
      func(m, n);
    }
  }

  template <typename M, typename N, typename Func>
  void farEach(Tenser<M>* a, Tenser<N>* b, Func func) {
    for (int i = 0; i < a->size(); i++) {
      M* m = &a->getData()[i];
      N* n = &b->getData()[i];
      func(m, n);
    }
  }
}  // namespace ForEach