//#pragma once
//#include <iostream>
//#include <vector>
//#include "cuda/Function.h"
//using namespace std;
//
//void TestCuda() {
//  creatDevice();
//
//  Tensorx<int> a({1, 2, 3, 4, 5});
//  Tensorx<int> b({10, 20, 30, 40, 55});
//  Tensorx<int> c(5);
//  add(c, a, b);
//  printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c.data()[0], c.data()[1], c.data()[2], c.data()[3], c.data()[4]);
//
//  Tensorx<double> d(vector<double>({0.6354469361189982}));
//  Tensorx<double> e(1);
//  sigmoid(d, e);
//  printf("%f\n", e.data()[0]);
//
//  Tensorx<double> f({0.6354469361189982,0.6354469361189982,0.6354469361189982,0.6354469361189982});
//  Tensorx<double> g({0.6354469361189982,0.6354469361189982,0.6354469361189982,0.6354469361189982});
//  Tensorx<double> k(4);
//  matmul(k, f, g, 2, 2, 2);
//  printf("%f\n", k.data()[0]);
//
//  cudaSynchronize();
//  cudaReset();
//}