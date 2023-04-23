#pragma once
#include <iostream>
#include "../cuda/Cuda.cu"

using namespace nvstd;

class Application;

using Func = nvstd::function<void(double)>;

class Context {
 public:
  __device__ double getValue() { return 0; }

  __device__ void setValue(double value) {}

  __device__ double getGrad() { return 0; }

  __device__ Application* setGrad(double grad) { return NULL; }
};

class Function {
 public:
  __device__ void compute() {}

  __device__ void gradient(){};
};

class Application : Context, Function {
 public:
  __device__ Application() {}

  __device__ Application(Context* context, double value) {
    this->value = value;
  }

  __device__ Application(Context* context, double value, Func inGrad1) {
    this->inGrad1 = inGrad1;
    this->value = value;
  }

  __device__ Application(Context* context, double value, Func inGrad1) {
    this->inGrad1 = inGrad1;
    this->value = value;
  }

  __device__ double getValue() { return value; }

  __device__ void setValue(double value) { this->value = value; }

  __device__ double getGrad() { return grad; }

  __device__ Application* setGrad(double grad) {
    this->grad = grad;
    return this;
  }

  __device__ void gradient() {
    if (inGrad1 != NULL) {
      inGrad1(grad);
    }
  }

 private:
  int idx = 0;
  double value = 0, grad = 0;
  Func inGrad1 = NULL;
};

class Operator : public Context, Function {
 public:
  __device__ Application* add(Application* cxt1, Application* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = valx + valy;

    Func grad1 = [cxt1, cxt2](double grad) {
      cxt1->setGrad(grad)->gradient();
      cxt2->setGrad(grad)->gradient();
    };
    return new Application(this, value, grad1);
  }

  __device__ Application* mul(Application* cxt1, Application* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = valx * valy;

    Func grad1 = [cxt1, cxt2, valx, valy](double grad) {
      cxt1->setGrad(grad * valy)->gradient();
      cxt2->setGrad(grad * valx)->gradient();
    };
    return new Application(this, value, grad1);
  }

  __device__ Application* minus(Application* cxt1, Application* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = valx - valy;

    Func grad1 = [cxt1, cxt2](double grad) {
      cxt1->setGrad(grad)->gradient();
      cxt2->setGrad(-grad)->gradient();
    };
    return new Application(this, value, grad1);
  }

  __device__ Application* powx(Application* cxt1, Application* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = pow(valx, valy);

    Func grad1 = [cxt1, valy, valx](double grad) {
      cxt1->setGrad(grad * valy * pow(valx, valy - 1))->gradient();
    };
    return new Application(this, value, grad1);
  }

  __device__ Application* var(double d) { return new Application(this, d); }

  __device__ Application* none(double d) { return new Application(this, d); }
};

class Square : public Operator {
 public:
  __device__ void compute() {
    Application* c = mul( none(0.5), powx(minus(none(0.01), var(0.391249035007275)), none(2)));
    c->setGrad(1)->gradient();
  }
};