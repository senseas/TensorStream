#pragma once
#include <iostream>
#include "../cuda/Cuda.cu"

using namespace nvstd;

class Application;
class Operator;

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

  __device__ double getValue() { return value; }

  __device__ void setValue(double value) { this->value = value; }

  __device__ double getGrad() { return grad; }

  __device__ void gradx(double grad) {}

  __device__ Application* setGrad(double grad) {
    this->grad = grad;
    return this;
  }

  __device__ void gradient() {
    if (cxt1 != NULL) {
      cxt1->gradx(grad);
    }
    if (cxt2 != NULL) {
      cxt2->gradx(grad);
    }
  }

 public:
  int idx = 0;
  double value = 0, grad = 0;
  Application *cxt1, *cxt2;
};



class AddGradient : public Application {
 public:
  __device__ AddGradient(Operator* context, double value, Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;

    printf("x={%f}\n", value);
  }
  
  __device__ void gradx(double grad) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    cxt1->setGrad(grad)->gradient();
    cxt2->setGrad(grad)->gradient();
  }
};

class MulGradient : public Application {
 public:
  __device__ MulGradient(Operator* context, double value, Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradx(double grad) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    cxt1->setGrad(grad * valy)->gradient();
    cxt2->setGrad(grad * valx)->gradient();
  }
};

class MinusGradient : public Application {
 public:
  __device__ MinusGradient(Operator* context, double value, Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradx(double grad) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    cxt1->setGrad(grad)->gradient();
    cxt2->setGrad(-grad)->gradient();
  }
};

class PowGradient : public Application {
 public:
  __device__ PowGradient(Operator* context, double value, Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradx(double grad) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    cxt1->setGrad(grad * valy * pow(valx, valy - 1))->gradient();
  }
};

class Operator : public Context, Function {
 public:
  __device__ Application* add(Application* cxt1, Application* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = valx + valy;
    return new AddGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* mul(Application* cxt1, Application* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = valx * valy;
    return new MulGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* minus(Application* cxt1, Application* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = valx - valy;
    return new MinusGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* powx(Application* cxt1, Application* cxt2) {
    double valx = cxt1->getValue(), valy = cxt2->getValue();
    double value = pow(valx, valy);
    return new PowGradient(this, value, cxt1);
  }

  __device__ Application* var(double d) { return new Application(this, d); }

  __device__ Application* none(double d) { return new Application(this, d); }
};

class Square : public Operator {
 public:
  __device__ void compute() {
    Application* c = mul(none(0.5), powx(minus(none(0.01), var(0.391249035007275)), none(2)));
    c->setGrad(1)->gradient();
    printf("x={%i}\n", c->getValue());
  }
};