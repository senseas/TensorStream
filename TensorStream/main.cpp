//#pragma once
//#include <iostream>
//#include <vector>
//#include "cuda/CudaUtil.h"
//#include "cuda/cudax.cu"
//#include "cuda/functionx.h"
//
//using namespace std;
//
//int main() {
//  creatDevice();
//  vector<int> a = {1, 2, 3, 4, 5};
//  vector<int> b = {10, 20, 30, 40, 55};
//  vector<int> c(5);
//  add(c, a, b);
//  printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);
//
//  vector<double> d = {0.6354469361189982};
//  vector<double> e = {0};
//  sigmoid(d, e);
//  printf("%f\n", e[0]);
//
//  vector<double> f = {0.6354469361189982,0.6354469361189982,0.6354469361189982,0.6354469361189982,};
//  vector<double> g = {0.6354469361189982,0.6354469361189982,0.6354469361189982,0.6354469361189982};
//  vector<double> k(4);
//  matmul(k, f, g, 2, 2, 2);
//  printf("%f\n", k[0]);
//
//  cudaSynchronize();
//
//  cudaReset();
//  return 0;
//}