#include "device_launch_parameters.h"
#include <stdio.h>

template<typename T>
__global__ void addKernel(T* c, const T* a, const T* b) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

template<typename T>
__global__ void sigmoidKernel(T* inx, T* out) {
  int i = threadIdx.x;
  out[i] = 1 / (1 + exp(inx[i]));
  printf("%f\n", out[0]);
}

template<typename T>
__global__ void matmulKernel(T* c, T* a, T* b, unsigned int h, unsigned int w, unsigned int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;

  //printf("x={%i}\n", idx);
  //printf("y={%i}\n", idx);

  T sum = 0;
  for (size_t i = 0; i < w; i++){
	  sum += a[idy * w + i] * b[i * n + idx];
  }

  c[idy * n + idx] = sum;
}
