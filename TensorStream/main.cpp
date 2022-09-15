#pragma once
#include <iostream>
#include <vector>
#include "cuda/CudaUtil.h"
#include "cuda/cudax.cu"
#include "cuda/Tensorx.h"

using namespace std;

// Helper function for using CUDA to add vectors in parallel.
template<typename T>
void add(vector<T>& c, vector<T>& a, vector<T>& b) {
  Tensor<T>* dev_a = new Tensor<T>(a);
  Tensor<T>* dev_b = new Tensor<T>(b);
  Tensor<T>* dev_c = new Tensor<T>(c);
  // Launch a kernel on the GPU with one thread for each element.
  addKernel << <1, c.size() >> > (dev_c->datax, dev_a->datax, dev_b->datax);
  getCudaDate(c, dev_c->datax);
}

// Helper function for using CUDA to add vectors in parallel.
template<typename T>
void sigmoid(vector<T>& a, vector<T>& b) {
  Tensor<T>* dev_a = new Tensor<T>(a);
  Tensor<T>* dev_b = new Tensor<T>(b);
  // Launch a kernel on the GPU with one thread for each element.
  sigmoidKernel << <1, 1 >> > (dev_a->datax, dev_b->datax);
  getCudaDate(b, dev_b->datax);
}

int main() {
  cudaError_t cudaStatus;
  vector<int> a = {1, 2, 3, 4, 5};
  vector<int> b = {10, 20, 30, 40, 55};
  vector<int> c(5);
  creatDevice();
  add(c, a, b);
  printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

  vector<double> d = {0.6354469361189982};
  vector<double> e = {0};
  sigmoid(d, e);
  printf("%f\n", e[0]);

  cudaSynchronize();
  cudaReset();

  return 0;
}