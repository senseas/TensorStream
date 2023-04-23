#pragma once
#include <iostream>
#include "../cuda/Cuda.cu"
namespace Math {
  __device__ double sin_(double a) {
    return sin(a);
  }

  __device__ double cos_(double a) {
    return cos(a);
  }

  __device__ double tan_(double a) {
    return tan(a);
  }

  __device__ double asin_(double a) {
    return asin(a);
  }

  __device__ double acos_(double a) {
    return acos(a);
  }

  __device__ double atan_(double a) {
    return atan(a);
  }

  __device__ double exp_(double a) {
    return exp(a);
  }

 __device__ double log_(double a) {
    return log(a);
  }

  __device__ double log10_(double a) {
    return log10(a);
  }

  __device__ double sqrt_(double a) {
    return sqrt(a);
  }

  __device__ double cbrt_(double a) {
    return cbrt(a);
  }

  __device__ double ceil_(double a) {
    return ceil(a);
  }

  __device__ double floor_(double a) {
    return floor(a);
  }

  __device__ double rint_(double a) {
    return rint(a);
  }

  __device__ double atan2_(double y, double x) {
    return atan2(y, x);
  }

  __device__ double pow_(double a, double b) {
    return pow(a, b);
  }
  __device__ double max_(double a, double b) {
    return a > b ? a : b;
  }
  __device__ double min_(double a, double b) {
    return a < b ? a : b;
  }
}  // namespace Math