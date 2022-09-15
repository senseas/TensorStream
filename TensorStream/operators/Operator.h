#pragma once
#include "../lang/Include.h"
#include "../lang/Objects.h"

using namespace Objects;

class AddOperator : public TensorOperator {
 public:
  AddOperator(Tensor* a, Tensor* b) : TensorOperator("Add", {a, b}){};

  Object compute() {
    double value = 0;
    for (int i = 0; i < getInput().size(); i++) {
      value += getInput<None*>(i)->value;
    }
    return new None(value);
  }

  void gradient() {
    None* out = getOutput<None*>();
    for (int i = 0; i < getInput().size(); i++) {
      getInput<None*>(i)->setGrad(out->grad);
    }
  }
};

class AddxOperator : public TensorOperator {
 public:
  AddxOperator(Tensor* a, Tensor* b) : TensorOperator("Addx", {a, b}){};

  Object compute() {
    Tenser<None*>* B = zeroNones(getInput<Tenser<None*>*>(0)->shape);
    for (int i = 0; i < getInput().size(); i++) {
      Tenser<None*>* A = getInput<Tenser<None*>*>(i);
      forEach(A, B, [](None* a, None* b) { b->value += a->value; });
    }
    return B;
  }

  void gradient() {
    Tenser<None*>* B = getOutput<Tenser<None*>*>();
    for (int i = 0; i < getInput().size(); i++) {
      Tenser<None*>* A = getInput<Tenser<None*>*>(i);
      forEach(A, B, [](None* a, None* b) { a->setGrad(b->grad); });
    }
  }
};

class SumOperator : public TensorOperator {
 public:
  SumOperator(Tensor* a) : TensorOperator("Sum", {a}){};

  Object compute() {
    Tenser<None*>* A = getInput<Tenser<None*>*>(0);
    None* b = new None(0);
    forEach(A, [b](None* a) { b->value += a->value; });
    return b;
  }

  void gradient() {
    Tenser<None*>* A = getInput<Tenser<None*>*>(0);
    None* b = getOutput<None*>();
    forEach(A, [b](None* a) { a->setGrad(b->grad); });
  }
};

class MinusOperator : public TensorOperator {
 public:
  template <typename... M>
  MinusOperator(Tensor* a, Tensor* b) : TensorOperator("Minus", {a, b}){};

  Object compute() {
    None *inx = getInput<None*>(0), *iny = getInput<None*>(1);
    double valx = inx->value, valy = iny->value;
    return new None(valx - valy);
  }

  void gradient() {
    None *inx = getInput<None*>(0), *iny = getInput<None*>(1);
    None* out = getOutput<None*>();
    double grad = out->grad;
    inx->setGrad(grad);
    iny->setGrad(-grad);
  }
};

class MinusxOperator : public TensorOperator {
 public:
  MinusxOperator(Tensor* a) : TensorOperator("Minus", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(-valx);
  }

  void gradient() {
    None *inx = getInput<None*>(0), *out = getOutput<None*>();
    double grad = out->grad;
    inx->setGrad(-grad);
  }
};

class MulOperator : public TensorOperator {
 public:
  MulOperator(Tensor* a, Tensor* b) : TensorOperator("Mul", {a, b}){};

  Object compute() {
    None *inx = getInput<None*>(0), *iny = getInput<None*>(1);
    double valx = inx->value, valy = iny->value;
    return new None(valx * valy);
  }

  void gradient() {
    None *inx = getInput<None*>(0), *iny = getInput<None*>(1);
    None* out = getOutput<None*>();

    double valx = inx->value, valy = iny->value;
    double grad = out->grad;
    inx->setGrad(grad * valy);
    iny->setGrad(grad * valx);
  }
};

class DivOperator : public TensorOperator {
 public:
  DivOperator(Tensor* a, Tensor* b) : TensorOperator("Div", {a, b}){};

  Object compute() {
    None *inx = getInput<None*>(0), *iny = getInput<None*>(1);
    double valx = inx->value, valy = iny->value;
    return new None(valx / valy);
  }

  void gradient() {
    None *inx = getInput<None*>(0), *iny = getInput<None*>(1);
    None* out = getOutput<None*>();

    double valx = inx->value, valy = iny->value;
    double grad = out->grad;
    inx->setGrad(grad / valy);
    iny->setGrad(-grad * valx / (valy * valy));
  }
};

class PowOperator : public TensorOperator {
 public:
  PowOperator(Tensor* a, Tensor* b) : TensorOperator("Pow", {a, b}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    None* iny = getInput<None*>(1);
    double valx = inx->value, valy = iny->value;
    return new None(pow(valx, valy));
  }

  void gradient() {
    None *inx = getInput<None*>(0), *iny = getInput<None*>(1);
    None* out = getOutput<None*>();
    double valx = inx->value, valy = iny->value;
    double grad = out->grad;
    inx->setGrad(grad * valy * pow(valx, valy - 1));
  }
};

class ExpOperator : public TensorOperator {
 public:
  ExpOperator(Tensor* a) : TensorOperator("Exp", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(exp(valx));
  }

  void gradient() {
    None *inx = getInput<None*>(0), *out = getOutput<None*>();
    double valx = inx->value;
    double grad = out->grad;
    inx->setGrad(grad * exp(valx));
  }
};

class ExpxFunction : public TensorFunction {
 public:
  ExpxFunction(Tensor* a) : TensorFunction("Expx", {a}){};

  Object compute() {
    Tenser<Tensor*>* A = getInput<Tenser<Tensor*>*>(0);
    Tenser<Tensor*>* B = zeroTensors(A->shape);
    farEach(A, B, [](Tensor** a, Tensor** b) { *b = new ExpOperator(*a); });
    return B;
  }
};

class LogOperator : public TensorOperator {
 public:
  LogOperator(Tensor* a) : TensorOperator("Log", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(log(valx));
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* out = getOutput<None*>();
    double valx = inx->value;
    double grad = out->grad;
    inx->setGrad(grad / valx);
  }
};

class MaxOperator : public TensorOperator {
 public:
  MaxOperator(Tensor* a, Tensor* b) : TensorOperator("Max", {a, b}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    None* iny = getInput<None*>(1);

    double valx = inx->value, valy = iny->value;
    return new None(max(valx, valy));
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* iny = getInput<None*>(1);
    None* out = getOutput<None*>();

    double valx = inx->value, valy = iny->value;
    double grad = out->grad;
    inx->setGrad(valx > valy ? grad : 0);
    iny->setGrad(valx < valy ? grad : 0);
  }
};

class MinOperator : public TensorOperator {
 public:
  MinOperator(Tensor* a, Tensor* b) : TensorOperator("Max", {a, b}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    None* iny = getInput<None*>(1);

    double valx = inx->value, valy = iny->value;
    return new None(min(valx, valy));
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* iny = getInput<None*>(1);
    None* out = getOutput<None*>();

    double valx = inx->value, valy = iny->value;
    double grad = out->grad;
    inx->setGrad(valx < valy ? 0 : grad);
    iny->setGrad(valx > valy ? 0 : grad);
  }
};

class SinOperator : public TensorOperator {
 public:
  SinOperator(Tensor* a) : TensorOperator("Sin", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(sin(valx));
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* out = getOutput<None*>();

    double valx = inx->value;
    double grad = out->grad;
    inx->setGrad(grad * cos(valx));
  }
};

class CosOperator : public TensorOperator {
 public:
  CosOperator(Tensor* a) : TensorOperator("Sin", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(cos(valx));
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* out = getOutput<None*>();

    double valx = inx->value;
    double grad = out->grad;
    inx->setGrad(-grad * sin(valx));
  }
};

class TanOperator : public TensorOperator {
 public:
  TanOperator(Tensor* a) : TensorOperator("Sin", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(tan(valx));
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* out = getOutput<None*>();

    double valx = inx->value;
    double grad = out->grad;
    inx->setGrad(grad / pow(cos(valx), 2));
  }
};

class CotOperator : public TensorOperator {
 public:
  CotOperator(Tensor* a) : TensorOperator("Sin", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(1 / tan(valx));
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* out = getOutput<None*>();

    double valx = inx->value;
    double grad = out->grad;
    inx->setGrad(-grad / pow(sin(valx), 2));
  }
};

class ArcsinOperator : public TensorOperator {
 public:
  ArcsinOperator(Tensor* a) : TensorOperator("Sin", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(asin(valx));
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* out = getOutput<None*>();

    double valx = inx->value;
    double grad = out->grad;
    inx->setGrad(grad / pow(1 - valx * valx, -2));
  }
};

class ArccosOperator : public TensorOperator {
 public:
  ArccosOperator(Tensor* a) : TensorOperator("Sin", {a}){};

  Object compute() {
    None* inx = getInput<None*>(0);
    double valx = inx->value;
    return new None(acos(valx));
  }

  void gradient() {
    None* inx = getInput<None*>(0);
    None* out = getOutput<None*>();

    double valx = inx->value;
    double grad = out->grad;
    inx->setGrad(-grad / pow(1 - valx * valx, -2));
  }
};