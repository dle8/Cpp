/*
    Tiled 1D convolution kernel using constant memory and general caching.
*/

#include <bits/stdc++.h>
using namespace std;

const int N = 7;
const int MASK_WIDTH = 5;
const int BLOCK_SIZE = 16;

__constant__ int dm[MASK_WIDTH];

void print(int *a, int sz) {
    for (int i = 0; i < sz; ++i) {
        cout << a[i] << " ";
    }
    cout << '\n';
}

void init(int *&a, int sz) {
    a = (int*) malloc(sz * sizeof(int));
    for (int i = 0; i < sz; ++i) {
        a[i] = rand() % 10;
    }
}

__global__ void convolution(int *da, int *dp, int mask_width, int width) {
    __shared__ int sa[N];

    // calculate dp[tx]
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    sa[threadIdx.x] = da[tx];

    __syncthreads();

    int tile_start = blockIdx.x * blockDim.x, tile_stop = (blockIdx.x + 1) * blockDim.x - 1;
    int pvalue = 0, start_idx = tile_start - mask_width / 2;

    for (int j = 0; j < mask_width; ++j) {
        int idx = start_idx + j;
        if (idx >= 0 && idx < width) { // if current index falls between the valid range
            if (idx >= tile_start && idx <= tile_stop) { // If in the current range of the current block, load from shared memory
                pvalue += sa[threadIdx.x - mask_width / 2 + j] * dm[j];
            } else { // else the value should be cached in L2
                pvalue += da[idx] * dm[j];
            }
        }
    }

    dp[tx] = pvalue;
}

int main() {
    int *ha, *hm, *hp, *da, *dp;
    init(ha, N);
    init(hm, MASK_WIDTH);
    init(hp, N);

    cudaMalloc(&da, N * sizeof(int));
    cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dp, N * sizeof(int));
    cudaMemcpyToSymbol(dm, hm, MASK_WIDTH * sizeof(int));

    dim3 block(BLOCK_SIZE);
    dim3 grid((N - 1) / BLOCK_SIZE + 1);

    convolution<<<grid, block>>>(da, dp, MASK_WIDTH, N);
    cudaMemcpy(hp, dp, N * sizeof(int), cudaMemcpyDeviceToHost);

    print(ha, N);
    print(hm, MASK_WIDTH);
    print(hp, N);

    free(ha);
    free(hm);
    free(hp);
    cudaFree(da);
    cudaFree(dm);
    cudaFree(dp);

    return 0;
}