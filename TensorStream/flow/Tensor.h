#pragma once
#include <iostream>
#include "../cuda/Cuda.cu"
#include "../flow/Math.h"
class Operator;

class Context {
 public:
  __device__ virtual double getValue() { return 0; }
  __device__ virtual void setValue(double value) {}
  __device__ virtual double getGrad() { return 0; }
  __device__ virtual void setGrad(double grad) {}
};

class Function {
 public:
  __device__ virtual void compute() {}
  __device__ virtual void gradient(double grad){};
};

class Application : public Context, public Function {
 public:
  __device__ Application() {}

  __device__ Application(Context* context, double value) {
    this->value = value;
  }

  __device__ virtual double getValue() { return value; }
  __device__ virtual void setValue(double value) { this->value = value; }
  __device__ virtual double getGrad() { return grad; }
  __device__ virtual void setGrad(double grad) { this->grad = grad; }

 public:
  int idx = 0;
  double value = 0, grad = 0;
  Application *cxt1 = NULL, *cxt2 = NULL;
};

class AddGradient : public Application {
 public:
  __device__ AddGradient(Operator* context, double value, 
  Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    cxt1->gradient(grad);
    cxt2->gradient(grad);
  }
};

class MinusGradient : public Application {
 public:
  __device__ MinusGradient(Operator* context, double value, 
  Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("MinusGradient={%f}\n", grad);
    cxt1->gradient(grad);
    cxt2->gradient(-grad);
  }
};

class MulGradient : public Application {
 public:
  __device__ MulGradient(Operator* context, double value, 
  Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("MulGradient={%f}\n", grad);
    double valx = cxt1->value, valy = cxt2->value;
    cxt1->gradient(grad * valy);
    cxt2->gradient(grad * valx);  
  }
};

class DivGradient : public Application {
 public:
  __device__ DivGradient(Operator* context, double value, 
  Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("DivGradient={%f}\n", grad);
    double valx = cxt1->value, valy = cxt2->value;
    cxt1->gradient(grad / valy);
    cxt2->gradient(grad * -valx / pow(valy, 2));  
  }
};

class ExpGradient : public Application {
 public:
  __device__ ExpGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("ExpGradient={%f}\n", grad);
    cxt1->gradient(grad * value);
  }
};

class PowGradient : public Application {
 public:
  __device__ PowGradient(Operator* context, double value, 
  Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("PowGradient={%f}\n", grad);
    double valx = cxt1->value, valy = cxt2->value;
    cxt1->gradient(grad * valy * pow(valx, valy - 1));
  }
};

class LogGradient : public Application {
 public:
  __device__ LogGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("LogGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad / valx);
  }
};

class SinGradient : public Application {
 public:
  __device__ SinGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("SinGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad * Math::cos_(valx));
  }
};

class CosGradient : public Application {
 public:
  __device__ CosGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("CosGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad * -Math::sin_(valx));
  }
};

class TanGradient : public Application {
 public:
  __device__ TanGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("TanGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad * Math::pow_(1 / Math::cos_(valx), 2));
  }
};

class CotGradient : public Application {
 public:
  __device__ CotGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("CotGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad * -Math::pow_(1 / Math::sin_(valx), 2));
  }
};

class SecGradient : public Application {
 public:
  __device__ SecGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("SecGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad * Math::tan_(valx) / Math::cos_(valx));
  }
};

class CscGradient : public Application {
 public:
  __device__ CscGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("CscGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad * -Math::cos_(valx) / Math::pow_(Math::sin_(valx), 2));
  }
};

class ArcsinGradient : public Application {
 public:
  __device__ ArcsinGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("ArcsinGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad / Math::pow_(1 - Math::pow_(valx, 2), -2));
  }
};

class ArccosGradient : public Application {
 public:
  __device__ ArccosGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("ArccosGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad / -Math::pow_(1 - Math::pow_(valx, 2), -2));
  }
};

class ArctanGradient : public Application {
 public:
  __device__ ArctanGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("ArctanGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad / (1 + Math::pow_(valx, 2)));
  }
};

class ArccotGradient : public Application {
 public:
  __device__ ArccotGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("ArccotGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad / -(1 + Math::pow_(valx, 2)));
  }
};

class ReluGradient : public Application {
 public:
  __device__ ReluGradient(Operator* context, double value, 
  Application* cxt1) {
    this->cxt1 = cxt1;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("ReluGradient={%f}\n", grad);
    double valx = cxt1->value;
    cxt1->gradient(grad * (valx > 0 ? 1 : 0.1));
  }
};

class MaxGradient : public Application {
 public:
  __device__ MaxGradient(Operator* context, double value, 
  Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("MaxGradient={%f}\n", grad);
    double valx = cxt1->value, valy = cxt2->value;
    cxt1->gradient(grad * (valx > valy ? 1 : 0));
    cxt2->gradient(grad * (valx < valy ? 1 : 0));
  }
};

class MinGradient : public Application {
 public:
  __device__ MinGradient(Operator* context, double value, 
  Application* cxt1, Application* cxt2) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    printf("MinGradient={%f}\n", grad);
    double valx = cxt1->value, valy = cxt2->value;
    cxt1->gradient(grad * (valx < valy ? 1 : 0));
    cxt2->gradient(grad * (valx > valy ? 1 : 0));
  }
};

class VarGradient : public Application {
 public:
  __device__ VarGradient(Operator* context, double value) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    //cxt1->setGrad(grad);
    printf("VarGradient={%f}\n", grad);
  }
};

class NoneGradient : public Application {
 public:
  __device__ NoneGradient(Operator* context, double value) {
    this->cxt1 = cxt1;
    this->cxt2 = cxt2;
    this->value = value;
  }

  __device__ void gradient(double grad) {
    //cxt1->setGrad(grad);
    printf("NoneGradient={%f}\n", grad);
  }
};

class Operator : public Context, public Function {
 public:
  __device__ Application* add(Application* cxt1, Application* cxt2) {
    double valx = cxt1->value, valy = cxt2->value;
    double value = valx + valy;
    return new AddGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* minus(Application* cxt1, Application* cxt2) {
    double valx = cxt1->value, valy = cxt2->value;
    double value = valx - valy;
    return new MinusGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* mul(Application* cxt1, Application* cxt2) {
    double valx = cxt1->value, valy = cxt2->value;
    double value = valx * valy;
    return new MulGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* div(Application* cxt1, Application* cxt2) {
    double valx = cxt1->value, valy = cxt2->value;
    double value = valx / valy;
    return new DivGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* exp(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::exp_(valx);
    return new ExpGradient(this, value, cxt1);
  }

  __device__ Application* pow(Application* cxt1, Application* cxt2) {
    double valx = cxt1->value, valy = cxt2->value;
    double value = Math::pow_(valx, valy);
    return new PowGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* log(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::log_(valx);
    return new LogGradient(this, value, cxt1);
  }

  __device__ Application* sin(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::sin_(valx);
    return new SinGradient(this, value, cxt1);
  }

  __device__ Application* cos(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::cos_(valx);
    return new CosGradient(this, value, cxt1);
  }

  __device__ Application* tan(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::tan_(valx);
    return new TanGradient(this, value, cxt1);
  }

  __device__ Application* cot(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::cos_(valx) / Math::sin_(valx);
    return new CotGradient(this, value, cxt1);
  }

  __device__ Application* sec(Application* cxt1) {
    double valx = cxt1->value;
    double value = 1 / Math::cos_(valx);
    return new SecGradient(this, value, cxt1);
  }

  __device__ Application* csc(Application* cxt1) {
    double valx = cxt1->value;
    double value = 1 / Math::sin_(valx);
    return new CscGradient(this, value, cxt1);
  }

  __device__ Application* arcsin(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::asin_(valx);
    return new ArcsinGradient(this, value, cxt1);
  }

  __device__ Application* arccos(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::acos_(valx);
    return new ArccosGradient(this, value, cxt1);
  }

  __device__ Application* arctan(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::atan_(valx);
    return new ArctanGradient(this, value, cxt1);
  }

  __device__ Application* arccot(Application* cxt1) {
    double valx = cxt1->value;
    double value = Math::atan_(1 / valx);
    return new ArccotGradient(this, value, cxt1);
  }

  __device__ Application* relu(Application* cxt1) {
    double valx = cxt1->value;
    double value = valx > 0 ? valx : 0.1 * valx;
    return new ReluGradient(this, value, cxt1);
  }

  __device__ Application* max(Application* cxt1, Application* cxt2) {
    double valx = cxt1->value, valy = cxt2->value;
    double value = Math::max_(valx, valy);
    return new MaxGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* min(Application* cxt1, Application* cxt2) {
    double valx = cxt1->value, valy = cxt2->value;
    double value = Math::min_(valx, valy);
    return new MinGradient(this, value, cxt1, cxt2);
  }

  __device__ Application* var(double d) { return new VarGradient(this, d); }

  __device__ Application* none(double d) { return new NoneGradient(this, d); }
};

class Square : public Operator {
 public:
  __device__ void compute() {
    Application* c = mul(none(0.5), pow(minus(none(0.01), var(0.391249035007275)), none(2)));
    c->gradient(1);
    printf("Square={%f}\n", c->value);
  }
};