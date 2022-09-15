#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

cudaError_t creatDevice() {
  cudaError_t cudaStatus;
  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	throw cudaStatus;
  }
  return cudaStatus;
}

cudaError_t cudaSynchronize() {
  cudaError_t cudaStatus;
  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	throw cudaStatus;
  }
  return cudaStatus;
};

cudaError_t cudaReset() {
  cudaError_t cudaStatus;
  // cudaDeviceReset must be called before exiting in order for profiling and
   // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceReset failed!");
	throw cudaStatus;
  }
  return cudaStatus;
};

template<typename T>
T* setCudaData(vector<T>& a) {
  T* dev_a = 0;
  cudaError_t cudaStatus;
  // Allocate GPU buffers for three vectors (two input, one output)    .
  cudaStatus = cudaMalloc((void**)&dev_a, a.size() * sizeof(T));
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(dev_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
  }
  return dev_a;
Error:
  cudaFree(dev_a);
  return dev_a;
}

template<typename T>
cudaError_t getCudaDate(vector<T>& c, T* dev_c) {
  cudaError_t cudaStatus;
  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	throw cudaStatus;
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(c.data(), dev_c, c.size() * sizeof(T), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	throw cudaStatus;
  }
  return cudaStatus;
};