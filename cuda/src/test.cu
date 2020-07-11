// #include <bits/stdc++.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

const int SECTION_SIZE = 2048;

__global__ void Brent_Kung_scan_kernel(float *X, float *Y, int InputSize) {
    __shared__ float XY[SECTION_SIZE];
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < InputSize) XY[threadIdx.x] = X[i];
    if (i + blockDim.x < InputSize) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) { 
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride -1;
        if (index < SECTION_SIZE) {
            XY[index] += XY[index - stride]; 
        }
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) { 
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index]; 
        }
    }
    __syncthreads();
    if (i < InputSize) Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < InputSize) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

int main() {
    int n = 1000000;
    float *dX, *dY, *hY, *hX;
    // cin >> n;
    size_t bytes = n * sizeof(float);
    hX = (float*) malloc(bytes);

    for (int i = 0; i < n; ++i) hX[i] = 1;
    hY = (float*) malloc(bytes);
    cudaMalloc((void**) &dX, bytes);
    cudaMalloc((void**) &dY, bytes);
    cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice);

    dim3 grid((n - 1) / 1024 + 1, 1, 1);
    dim3 block(1024, 1, 1); // (256, 1, 1)?
    Brent_Kung_scan_kernel<<<grid, block, 1024* sizeof(float)>>>(dX, dY, n);
    
    cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < n; ++i) cout << hY[i] << ' ';
    cout << hY[2048];
    
    free(hX);
    free(hY);
    cudaFree(dX);
    cudaFree(dY);
    return 0;
}