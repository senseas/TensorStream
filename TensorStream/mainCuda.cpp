#pragma once
#include <iostream>
#include <vector>
#include "cuda/Cuda.cu"
#include "flow/Tensor.h"
using namespace std;

int main() {
  cudaForEach<<<1000000, 10>>>(10000000, [] __device__(int i) {
    Square* aaa= new Square();
    aaa->compute();
    printf("index={%i}\n", i);
  });
  return 0;
}