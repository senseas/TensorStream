#pragma once

#include "../lang/ForEach.h"
#include "../lang/Objects.h"
#include "../lang/Tenser.h"
#include "../core/None.h"
#include "../core/Tensor.h"

using namespace ForEach;

namespace TensorFlux {

  static const double EX = 0.0000000001;

  Object getOutput(Object& o) {
    if (o.type() == typeid(Tenser<Tensor*>*)) {
      Tenser<Tensor*>* a = o.get<Tenser<Tensor*>*>();
      Tenser<Object*> b(a->shape);
      farEach(a, &b, [](Tensor** m, Object** n) { *n = &(*m)->getOutput(); });
      vector<int> shape = Objects::shapes(&b);
      Tenser<None*>* c = new Tenser<None*>(shape);
      farEach(&b, c, [](Object** m, None** n) { *n = (*m)->get<None*>(); });
      b.clear();
      return c;
    }
    else {
      Tensor* a = o.get<Tensor*>();
      return a->getOutput();
    }
  }

  void createOutput(Tensor* tensor, vector<int> shape) {
    if (tensor->getOutput().isNull()) {
      tensor->shape = shape;
      tensor->value = Objects::listof(shape, 0.0);
      tensor->grad = Objects::listof(shape, 0.0);
      tensor->reduce = Objects::listof(shape, false);
      tensor->output = Objects::zeroNones(tensor, false);
    }
  }

  void createOutput(Tensor* tensor) {
    if (tensor->getOutput().isNull()) {
      tensor->value = new double[1] {0.0};
      tensor->grad = new double[1] {0.0};
      tensor->reduce = new bool[1] {false};
      tensor->output = zeroNone(tensor, false);
    }
  }

  void createOutput(Tensor* tensor, Object& o) {
    if (Objects::isTenser(o)) {
      createOutput(tensor, o.get<Tenser<None*>*>()->shape);
    }
    else {
      createOutput(tensor);
    }
  }

  void forwards(Tensor* tensor) {
    if (tensor->funcout.isNull()) {
      tensor->funcout = getOutput(tensor->function);
      createOutput(tensor, tensor->funcout);
    }
    forEach<None*>(tensor->getOutput(), tensor->funcout, [](None* out, None* none) {
      out->setValue(none->getValue());
      out->reset();
    });
  }

  void forward(Tensor* tensor) {
    Object& funcs = tensor->getFunction();
    forEach<Tensor*>(funcs, [](Tensor* a) { a->forward(); });
    forwards(tensor);
  }

  void backwards(Tensor* tensor) {
    forEach<None*>(tensor->getOutput(), tensor->funcout, [](None* out, None* none) {
      none->setGrad(out->getGrad());
    });
  }

  void backward(Tensor* tensor) {
    backwards(tensor);
    Object& funcs = tensor->getFunction();
    forEach<Tensor*>(funcs, [](Tensor* a) { a->backward(); });
  }

  void reduce(Tensor* tensor) {
    Object& funcs = tensor->getFunction();
    forEach<Tensor*>(funcs, [](Tensor* a) { a->reducer(); });
  }

  void compute(Tensor* tensor) {
    Object nones = tensor->compute();
    Object outs = tensor->getOutput();
    if (outs.nonNull()) {
      forEach<None*>(outs, nones, [](None* out, None* none) {
        out->reset();
        out->setValue(none->getValue());
      });
      if (Objects::isTenser(nones)) {
        Tenser<None*>* m = nones.get<Tenser<None*>*>();
        for (int i = 0; i < m->size(); i++) {
          delete m->getData()[i];
        }
        m->clear();
        delete m;
      }
      else {
        None* m = nones.get<None*>();
        delete m;
      }
    }
    else {
      tensor->output = nones;
    }
  }

  void computer(Tensor* tensor) {
    if (Objects::isNone(tensor)) {
      Object& outs = tensor->getOutput();
      forEach<None*>(outs, [](None* out) { out->reset(); });
    }
    else {
      tensor->forward();
    }
  }

  void gradient(Tensor* tensor) {
    tensor->gradient();
  }

  void reducer(Tensor* tensor) {
    if (Objects::isNone(tensor)) {
      Object& outs = tensor->getOutput();
      forEach<None*>(outs, [](None* none) {
        if (none->isGrad && !none->getReduce()) {
          none->setReduce(true);
          // double valu = abs(none->value), grad = abs(none->grad);
          // double rate = min(valu / (grad + EX), grad / (valu + EX)) * 0.003;
          // double value = none->value - rate * none->grad;
          none->setValue(none->getValue() - 0.3 * none->getGrad());
        }
      });
    }
    else {
      tensor->reducer();
    }
  }

  template <typename M>
  M getTensor(Object& o) {
    if (Objects::isTenser(o)) {
      Tenser<None*>* a = o.get<Tenser<None*>*>();
      Tenser<Tensor*>* b = new Tenser<Tensor*>(a->shape);
      farEach(a, b, [](None** c, Tensor** d) { *d = new Tensor(*c); });
      return (M)b;
    }
    else {
      None* a = o.get<None*>();
      return (M) new Tensor(a);
    }
  }

};  // namespace TensorFlux
