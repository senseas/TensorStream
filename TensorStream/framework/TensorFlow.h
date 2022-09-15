#pragma once

#include "../operators/MatmulOperator.h"
#include "../operators/Operator.h"
#include "../operators/ReluOperator.h"
#include "../operators/ShapeFunction.h"
#include "../operators/SigmoidFunction.h"
#include "../operators/SquareFunction.h"
//#include "../operators/LossFunction.h"
//#include "../operators/SoftmaxCrossFunction.h"

namespace Flow {

Tensor* add(Tensor* a, Tensor* b) {
  return new AddOperator(a, b);
}

Tensor* addx(Tensor* a, Tensor* b) {
  return new AddxOperator(a, b);
}

Tensor* sum(Tensor* a) {
  return new SumOperator(a);
}

Tensor* minus(Tensor* a, Tensor* b) {
  return new MinusOperator(a, b);
}

Tensor* minus(Tensor* a) {
  return new MinusxOperator(a);
}

Tensor* mul(Tensor* a, Tensor* b) {
  return new MulOperator(a, b);
}

Tensor* div(Tensor* a, Tensor* b) {
  return new DivOperator(a, b);
}

Tensor* exp(Tensor* a) {
  return new ExpOperator(a);
}

Tensor* expx(Tensor* a) {
  return new ExpxFunction(a);
}

Tensor* pow(Tensor* a, Tensor* b) {
  return new PowOperator(a, b);
}

Tensor* log(Tensor* a) {
  return new LogOperator(a);
}

// Tensor* prod(Tensor* a, Tensor* b) {
//   return new ProdFunction(a, b);
// }

Tensor* relu(Tensor* a) {
  return new ReluOperator(a);
}

Tensor* relux(Tensor* a) {
  return new ReluxOperator(a);
}

Tensor* max(Tensor* a, Tensor* b) {
  return new MaxOperator(a, b);
}

Tensor* matmul(Tensor* a, Tensor* b) {
  return new MatmulOperator(a, b);
}

// Tensor* matmulTran(Tensor* a, Tensor* b) {
//   return new MatmulTranOperator(a, b);
// }

Tensor* shape(Tensor* a, Tensor* b) {
  return new ShapeFunction(a, b);
}

Tensor* sigmoid(Tensor* a) {
  return new SigmoidFunction(a);
}

Tensor* sigmoidx(Tensor* a) {
  return new SigmoidxFunction(a);
}

// Tensor* square(Tensor* a, Tensor* b) {
//   return new SquareFunction(a, b);
// }

Tensor* squarex(Tensor* a, Tensor* b) {
  return new SquarexFunction(a, b);
}

// Tensor* softmaxCross(Tensor* a, Tensor* b) {
//   return new SoftmaxCrossFunction(a, b);
// }

// Tensor* softmaxCrossx(Tensor* a, Tensor* b) {
//   return new SoftmaxCrossxFunction(a, b);
// }

// Tensor* sigmoidCross(Tensor* a, Tensor* b) {
//   return new SigmoidCrossFunction(a, b);
// }

// Tensor* sigmoidCrossx(Tensor* a, Tensor* b) {
//   return new SigmoidCrossxFunction(a, b);
// }

// Tensor*conv(List<int> stride, int padding, Tensor* a, Tensor* b) {
//   return new ConvOperator(stride, padding, a, b);
// }

// Tensor*convx(List<int> stride, int padding, Tensor* a, Tensor* b) {
//   return new ConvxFunction(stride, padding, a, b);
// }

// Tensor*deconv(List<int> stride, int padding, Tensor* a, Tensor* b) {
//   return new DeconvOperator(stride, padding, a, b);
// }

// Tensor*deconvx(List<int> stride, int padding, Tensor* a, Tensor* b) {
//   return new DeconvxFunction(stride, padding, a, b);
// }

// Tensor*maxpool(int kernelSize, List<int> stride, int padding, Tensor* a,
// Tensor* b) {
//   return new MaxpoolOperator(kernelSize, stride, padding, a, b);
// }

// Tensor*maxpoolx(int kernelSize, List<int> stride, int padding, Tensor* a,Tensor* b) {
//   return new MaxpoolxFunction(kernelSize, stride, padding, a, b);
// }

// Tensor*demaxpool(int kernelSize, List<int> stride, int padding, Tensor* a,
// Tensor* b) {
//   return new DemaxpoolOperator(kernelSize, stride, padding, a, b);
// }

// Tensor*demaxpoolx(int kernelSize, List<int> stride, int padding, Tensor* a,
// Tensor* b) {
//   return new DemaxpoolxFunction(kernelSize, stride, padding, a, b);
// }

// Tensor*softmax(Tensor* a, Tensor* b) {
//   return new SoftmaNFunction(a, b);
// }

// Tensor*selfAttention(Tensor* a, Tensor* b) {
//   return new SelfAttentionFunction(a, b);
// }

// Tensor*batchNorm(Tensor* a, Tensor* b) {
//   return new BatchNormFunction(a, b);
// }
}  // namespace Flow
