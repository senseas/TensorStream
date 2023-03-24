#pragma once
#include "device_launch_parameters.h"
#include <stdio.h>

template <typename T> 
__global__ void addKernel(T *c, const T *a, const T *b) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

template <typename T> 
__global__ void sigmoidKernel(T *inx, T *out) {
  int i = threadIdx.x;
  out[i] = 1 / (1 + exp(inx[i]));
  // printf("%f\n", out[0]);
}

template <typename T>
__global__ void matmulKernel(T *a, T *b, T *c, int h, int w, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;

  // printf("x={%i}\n", idx);
  // printf("y={%i}\n", idy);

  T sum = 0;
  for (size_t i = 0; i < n; i++) {
    sum += a[idx * n + i] * b[i * w + idy];
  }

  c[idy * n + idx] = sum;
}

template <typename Func> 
__global__ void cudaForEach1D(size_t N, Func func) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (N); i += blockDim.x * gridDim.x) {
    func(i);
  }
}

template <typename Func>
__global__ void cudaForEach2D(size_t M, size_t N, Func func) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (M); i += blockDim.x * gridDim.x) {
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (N); j += blockDim.y * gridDim.y) {
      func(i, j);
    }
  }
}