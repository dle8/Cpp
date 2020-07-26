#include <bits/stdc++.h>
#include <iostream>
#include <stdio.h>

using namespace std;

// histogram is in global memory -> contention within grid, or local memory
/*
__global__ void history_kernel(char* dx, int size, int* histogram) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        int pos = dx[i] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&histogram[pos / 4], 1);
        }
    }
}
*/

__global__ void inside_kernel(char *dx, int size, int* histogram) {

}

// histogram_private is in shared memory -> contention now only limits within block, or shared memory
__global__ void history_kernel(char *dx, int size, int* histogram) {
    __shared__ int histogram_private[7];
    if (threadIdx.x < 7) histogram_private[threadIdx.x] = 0;
    __syncthreads();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        int pos = dx[i] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&histogram_private[pos / 4], 1);
        }
    }

    if (threadIdx.x < 7) atomicAdd(&histogram[threadIdx.x], histogram_private[threadIdx.x]);
}

int main() {
    char *dx, *str;
    int n; 
    scanf("%d", &n);
    str = (char*) malloc(n * sizeof(char));
    scanf("%s", str);
    int *dh, *hh;
    size_t sz = strlen(str) * sizeof(char);
    hh = (int*) calloc(7, sizeof(int));

    cudaMalloc((void**) &dx, sz);
    cudaMalloc((void**) &dh, 7 * sizeof(int));
    cudaMemcpy(dx, str, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dh, hh, 7 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim(256);
    dim3 grid_dim(((int)sz - 1) / 256 + 1);

    history_kernel<<<grid_dim, block_dim>>>(dx, (int)sz, dh);

    cudaMemcpy(hh, dh, 7 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i <7; ++i) cout << i << " " << hh[i] << '\n';
    free(hh);
    cudaFree(dh);
    cudaFree(dx);
    return 0;
}