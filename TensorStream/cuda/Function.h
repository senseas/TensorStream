#pragma once
#include "../cuda/CudaUtil.h"
#include "../cuda/Cuda.cu"
#include "../cuda/Tensorx.h"
using namespace std;

// Helper function for using CUDA to add vectors in parallel.
template<typename T>
void add(Tensorx<T>& c, Tensorx<T>& a, Tensorx<T>& b) {
  // Launch a kernel on the GPU with one thread for each element.
  addKernel << <1, c.size() >> > (c.datax(), a.datax(), b.datax());
  getCudaDate(c);
}

// Helper function for using CUDA to add vectors in parallel.
template<typename T>
void sigmoid(Tensorx<T>& a, Tensorx<T>& b) {
  // Launch a kernel on the GPU with one thread for each element.
  sigmoidKernel << <1, 1 >> > (a.datax(), b.datax());
  getCudaDate(b);
}

// Helper function for using CUDA to add vectors in parallel.
template<typename T>
void matmul(Tensorx<T>& a, Tensorx<T>& b, Tensorx<T>& c, int h, int w, int n) {
  // Launch a kernel on the GPU with one thread for each element.
  matmulKernel << <h, w >> > (a.datax(), b.datax(), c.datax(), h, w, n);
  getCudaDate(c);
}