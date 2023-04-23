#pragma once
#include <math.h>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace std;

class Operator;

using Func = function<Operator*(double data)>;

class Context {
 public:
  double getValue() { return 0; }

  void setValue(double value) {}

  double getGrad() { return 0; }

  Operator* setGrad(double grad) { return NULL; }
};

class Function {
 public:
  void compute() {}

  void gradient(){};
};

class Operator : public Context, Function {
 public:
  Operator() {}

  Operator(Operator* context, double value) { this->value = value; }

  Operator(Operator* context, double value, Func inGrad1) {
    this->inGrad1 = inGrad1;
    this->value = value;
  }

  Operator(Operator* context, double value, Func inGrad1, Func inGrad2) {
    this->inGrad1 = inGrad1;
    this->inGrad2 = inGrad2;
    this->value = value;
  }

  Operator* add(Operator* cxt1, Operator* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = valx + valy;

    Func grad1 = [cxt1](double grad) { return cxt1->setGrad(grad); };
    Func grad2 = [cxt2](double grad) { return cxt2->setGrad(grad); };
    return new Operator(this, value, grad1, grad2);
  }

  Operator* mul(Operator* cxt1, Operator* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = valx * valy;

    Func grad1 = [cxt1, valy](double grad) { return cxt1->setGrad(grad * valy); };
    Func grad2 = [cxt2, valx](double grad) { return cxt2->setGrad(grad * valx); };
    return new Operator(this, value, grad1, grad2);
  }

  Operator* minus(Operator* cxt1, Operator* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = valx - valy;

    Func grad1 = [cxt1](double grad) { return cxt1->setGrad(grad); };
    Func grad2 = [cxt2](double grad) { return cxt2->setGrad(-grad); };
    return new Operator(this, value, grad1, grad2);
  }

  Operator* pow(Operator* cxt1, Operator* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = std::pow(valx, valy);

    Func grad1 = [cxt1, valy, valx](double grad) { return cxt1->setGrad(grad * valy * std::pow(valx, valy - 1)); };
    return new Operator(this, value, grad1);
  }

  Operator* var(double d) { return new Operator(this, d); }

  Operator* none(double d) { return new Operator(this, d); }

  double getValue() { return value; }

  void setValue(double value) { this->value = value; }

  double getGrad() { return grad; }

  Operator* setGrad(double grad) {
    this->grad = grad;
    return this;
  }

  void gradient() {
    if (inGrad1 != NULL) {
      inGrad1(grad)->gradient();
    }
    if (inGrad2 != NULL) {
      inGrad2(grad)->gradient();
    }
  }

 private:
  int idx = 0;
  double value = 0, grad = 0;
  Func inGrad1 = NULL, inGrad2 = NULL;
};

class Minus : public Operator {
 public:
  void compute() {
    Operator* bbb = var(0.391249035007275);
    Operator* sssss=none(0.01);
    Operator* aaa = mul(none(0.5), pow(minus(sssss, bbb), none(2)));
    aaa->setGrad(1)->gradient();
  }
};