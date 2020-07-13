#include <bits/stdc++.h>
using namespace std;

const int BLOCK_SIZE = 32;

__device__ void increase(float &a) {
    a += 1;
}

__global__ void kernel(float* da, int sz) {
    // int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // if (tid < sz) increase(da[tid]);
    if (threadIdx.x % 2 == 0) __syncthreads();
    else __syncthreads();
}

void print(float *a, int sz) {
    for (int i = 0; i < sz; ++i) cout << a[i] << ' ';
    cout << '\n';
}

int main() {
    float *ha, *da;
    int n = 1500;
    ha = (float*) malloc(n * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        ha[i] = rand() % 10;
    }
    cudaMalloc(&da, n * sizeof(float));
    cudaMemcpy(da, ha, n * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE);
    dim3 grid((n - 1) / BLOCK_SIZE + 1);

    kernel<<<grid, block>>>(da, n);

    print(ha, n);
    cudaMemcpy(ha, da, n * sizeof(float), cudaMemcpyDeviceToHost);
    print(ha, n);
    cudaFree(da);
    free(ha);

    return 0;
}