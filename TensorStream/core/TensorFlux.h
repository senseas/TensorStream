#pragma once

#include <memory>

#include "../core/None.h"
#include "../core/Tensor.h"
#include "../lang/ForEach.h"
#include "../lang/Objects.h"
#include "../lang/Tenser.h"

using namespace ForEach;
using namespace std;

namespace TensorFlux {

  static const double EX = 0.0000000001;

  Object getOutput(Object& o) {
    if (Objects::isTenser<Tensor>(o)) {
      shared_ptr<Tenser<Tensor*>> a = make_shared<Tenser<Tensor*>>(o.get<Tenser<Tensor*>*>());
      shared_ptr<Tenser<Object*>> b = make_shared<Tenser<Object*>>(a->shape);
      farEach(a, b, [](Tensor** m, Object** n) { *n = &(*m)->getOutput(); });
      vector<int> shape = Objects::shapes(b);
      shared_ptr<Tenser<None*>> c = make_shared<Tenser<None*>>(shape);
      farEach(b, c, [](Object** m, None** n) { *n = (*m)->get<None*>(); });
      return c;
    } else {
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
      tensor->value = new double[1]{0.0};
      tensor->grad = new double[1]{0.0};
      tensor->reduce = new bool[1]{false};
      tensor->output = zeroNone(tensor, false);
    }
  }

  void createOutput(Tensor* tensor, Object& o) {
    if (Objects::isTenser<None>(o)) {
      createOutput(tensor, o.get<shared_ptr<Tenser<None*>>>()->shape);
    } else {
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

  void resetOutput(Tensor* tensor) {
    if (Objects::nonNull(tensor->value)) {
      forEach(Objects::shapeSize(tensor->shape), [tensor](int i) {
        tensor->value[i] = 0.0;
        tensor->grad[i] = 0.0;
        tensor->reduce[i] = false;
      });
    }
  }

  void compute(Tensor* tensor) {
    Object outs = tensor->getOutput();
    if (outs.nonNull()) {
      resetOutput(tensor);
      Object nones = tensor->compute();
      if (Objects::eqTenser<None>(nones, outs)) {
        forEach<None*>(outs, nones, [](None* out, None* none) {
          out->setValue(none->getValue());
        });
      } else {
        forEach<None*>(outs, nones, [](None* out, None* none) {
          out->setValue(none->getValue());
          out->reset();
        });
        if (Objects::isTenser<None>(nones)) {
          shared_ptr<Tenser<None*>> m = nones.get<shared_ptr<Tenser<None*>>>();
          m->clear();
        } else {
          None* m = nones.get<None*>();
          delete m;
        }
      }
    } else {
      Object nones = tensor->compute();
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
        if (none->isGrad && !none->getReduce()) {
          none->setReduce(true);
          // double valu = abs(none->value), grad = abs(none->grad);
          // double rate = min(valu / (grad + EX), grad / (valu + EX)) * 0.003;
          // double value = none->value - rate * none->grad;
          none->setValue(none->getValue() - 0.3 * none->getGrad());
        }
      });
    } else {
      tensor->reducer();
    }
  }

  Object getTensor(Object& o) {
    if (Objects::isTenser<None>(o)) {
      shared_ptr<Tenser<None*>> a = o.get<shared_ptr<Tenser<None*>>>();
      shared_ptr<Tenser<Tensor*>> b = make_shared<Tenser<Tensor*>>(a->shape);
      farEach(a, b, [](None** c, Tensor** d) { *d = new Tensor(*c); });
      return b;
    } else {
      None* a = o.get<None*>();
      return new Tensor(a);
    }
  }

};  // namespace TensorFlux
