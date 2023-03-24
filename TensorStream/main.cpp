#pragma once
#include <iostream>
#include <vector>
#include "TestNN.h"
using namespace std;

int main() {
  creatDevice();
  NNTest();
  cudaReset();
  return 0;
}