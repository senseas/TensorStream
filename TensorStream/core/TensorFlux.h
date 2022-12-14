#pragma once

#include "../lang/ForEach.h"
#include "../lang/Objects.h"
#include "../lang/Tenser.h"
#include "../core/None.h"
#include "../core/Tensor.h"
using namespace ForEach;

namespace TensorFlux {

static const double EX = 0.0000000001;

void forward(Tensor* tensor) {
  Object& funcs = tensor->getFunction();
  forEach<Tensor*>(funcs, [](Tensor* a) { a->forward(); });
}

void backward(Tensor* tensor) {
  Object& funcs = tensor->getFunction();
  forEach<Tensor*>(funcs, [](Tensor* a) { a->backward(); });
}

void reduce(Tensor* tensor) {
  Object& funcs = tensor->getFunction();
  forEach<Tensor*>(funcs, [](Tensor* a) { a->reduce(); });
}

void compute(Tensor* tensor) {
  Object nones = tensor->compute();
  Object outs = tensor->getOutput();
  if (outs.nonNull()) {
    forEach<None*>(outs, nones, [](None* out, None* none) {
      out->reset();
      out->value = none->value;
    });
    if (nones.type() == typeid(Tenser<None*>*)) {
      Tenser<None*>* m = nones.get<Tenser<None*>*>();
      delete m;
    } else {
      None* m = nones.get<None*>();
      delete m;
    }
  } else {
    tensor->output = nones;
  }
}

void computer(Tensor* tensor) {
  if (Objects::isNone(tensor)) {
    Object& outs = tensor->getOutput();
    forEach<None*>(outs, [](None* out) { out->reset(); });
  } else {
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
      if (none->gradre && !none->reduce) {
        none->reduce = true;
        // double valu = abs(none->value), grad = abs(none->grad);
        // double rate = min(valu / (grad + EX), grad / (valu + EX)) * 0.003;
        // double value = none->value - rate * none->grad;
        none->value = none->value - 0.3 * none->grad;
      }
    });
  } else {
    tensor->reduce();
  }
}

Object getOutput(Object& o) {
  if (o.type() == typeid(Tenser<Tensor*>*)) {
    Tenser<Tensor*>* a = o.get<Tenser<Tensor*>*>();
    Tenser<Object*> b = Tenser<Object*>(a->shape);
    farEach(a, &b, [](Tensor** m, Object** n) { *n = &(*m)->getOutput(); });
    vector<int> shape = Objects::shapes(&b);
    Tenser<None*>* c = new Tenser<None*>(shape);
    farEach(&b, c, [](Object** m, None** n) { *n = (*m)->get<None*>(); });
    return c;
  } else {
    Tensor* a = o.get<Tensor*>();
    return a->getOutput();
  }
}

template <typename M>
M getTensor(Object& o) {
  if (o.type() == typeid(Tenser<None*>*)) {
    Tenser<None*>* a = o.get<Tenser<None*>*>();
    Tenser<Tensor*>* b = new Tenser<Tensor*>(a->shape);
    farEach(a, b, [](None** c, Tensor** d) { *d = new Tensor(*c); });
    return (M)b;
  } else {
    None* a = o.get<None*>();
    return (M) new Tensor(a);
  }
}

};  // namespace TensorFlux
