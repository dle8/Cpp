/*
    Tiled 1D convolution kernel using constant memory and general caching.

    Recall that the halo calls of a block are also internal cells of a neighboring block. So there is a signficant probability that by the time
    the next block needs to use these halo cells, they are already in the cache due to the access by the previous block. As a result, the memory
    accesses to these halo cells may be naturally served from cache without causing additional DRAM traffic. That is, we can leave the accesses
    to thse ahlo cells in the original N elements rather than loading into the N_ds. So now our kernel onlu loads the internal elements of each
    tile into the shared memory.
*/

#include <bits/stdc++.h>
using namespace std;

const int N = 7;
const int MASK_WIDTH = 5;
const int BLOCK_SIZE = 16;

__constant__ int M[MASK_WIDTH];

__global__ void convolution(int *N, int *P, int mask_width, int width) {
    __shared__ int N_ds[N];

    // calculate P[tx]
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    N_ds[threadIdx.x] = N[tx];

    __syncthreads();

    int tile_start = blockIdx.x * blockDim.x, tile_stop = (blockIdx.x + 1) * blockDim.x - 1;
    int pvalue = 0, start_idx = tile_start - mask_width / 2;

    for (int j = 0; j < mask_width; ++j) {
        int idx = start_idx + j;
        if (idx >= 0 && idx < width) { // if current index falls between the valid range
            if (idx >= tile_start && idx <= tile_stop) { // If in the current range of the current block, load from shared memory
                pvalue += N_ds[threadIdx.x - mask_width / 2 + j] * M[j];
            } else { // else the value should be cached in L2
                pvalue += N[idx] * M[j];
            }
        }
    }

    P[tx] = pvalue;
}

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

int main() {
    int *ha, *hm, *hp, *da, *dp;
    init(ha, N);
    init(hm, MASK_WIDTH);
    init(hp, N);

    cudaMalloc(&da, N * sizeof(int));
    cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dp, N * sizeof(int));
    cudaMemcpyToSymbol(M, hm, MASK_WIDTH * sizeof(int));

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
    cudaFree(M);
    cudaFree(dp);

    return 0;
}