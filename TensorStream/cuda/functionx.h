#pragma once
#include <iostream>
#include <vector>
#include "../cuda/CudaUtil.h"
#include "../cuda/cudax.cu"
#include "../cuda/Tensorx.h"

using namespace std;

// Helper function for using CUDA to add vectors in parallel.
template<typename T>
void add(vector<T>& c, vector<T>& a, vector<T>& b) {
  Tensorx<T>* dev_a = new Tensorx<T>(a);
  Tensorx<T>* dev_b = new Tensorx<T>(b);
  Tensorx<T>* dev_c = new Tensorx<T>(c);
  // Launch a kernel on the GPU with one thread for each element.
  addKernel << <1, c.size() >> > (dev_c->datax, dev_a->datax, dev_b->datax);
  getCudaDate(c, dev_c->datax);
}

// Helper function for using CUDA to add vectors in parallel.
template<typename T>
void sigmoid(vector<T>& a, vector<T>& b) {
  Tensorx<T>* dev_a = new Tensorx<T>(a);
  Tensorx<T>* dev_b = new Tensorx<T>(b);
  // Launch a kernel on the GPU with one thread for each element.
  sigmoidKernel << <1, 1 >> > (dev_a->datax, dev_b->datax);
  getCudaDate(b, dev_b->datax);
}

// Helper function for using CUDA to add vectors in parallel.
template<typename T>
void matmul(vector<T>& c, vector<T>& a, vector<T>& b, int h, int w, int n) {
  Tensorx<T>* dev_a = new Tensorx<T>(a);
  Tensorx<T>* dev_b = new Tensorx<T>(b);
  Tensorx<T>* dev_c = new Tensorx<T>(c);
  // Launch a kernel on the GPU with one thread for each element.
  matmulKernel << <h, w >> > (dev_c->datax, dev_a->datax, dev_b->datax, h, w, n);
  getCudaDate(c, dev_c->datax);
}