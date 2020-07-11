/*
    Tiled 1D convolution kernel using constant memory.

    Cost and benefit:
        - O_TILE_WIDTH+MASK_WIDTH -1 elements loaded for each input tile
        – O_TILE_WIDTH*MASK_WIDTH global memory accesses replaced by shared memory accesses
        - reduction factor of (O_TILE_WIDTH * MASK_WIDTH) / (O_TILE_WIDTH + MASK_WIDTH - 1)
*/

#include <bits/stdc++.h>
using namespace std;

const int N = 7;
const int MASK_WIDTH = 5;
const int BLOCK_SIZE = 16;

__constant__ int dm[MASK_WIDTH];

void print(int *a, int sz) {
    for (int i = 0; i < sz; ++i) cout << a[i] << " ";
    cout << '\n';
}

void init(int* &a, int sz) {
    a = (int*) malloc(sz * sizeof(int));
    srand (time(NULL));
    for (int i = 0; i < sz; ++i) {
        a[i] = rand() % 10;
    }
}

__global__ void convolution(int* da, int* dp, int mask_width, int width) {
    __shared__ int sa[N + MASK_WIDTH / 2];

    int n = mask_width / 2;

    // calculate dp[tx] 
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    // Make the last threads load the first hallo cells and make the first threads load the last hallo cells. This helps
    // reduce number of global memory loads -> each element needed in the input for the convolution is loaded once. In this
    // design, ome threads need to load more than one input element into the shared memory, and all threads participate 
    // in calculating output elements
    
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x > blockDim.x - n) { // Make the last n threads load the left hallo cells
        sa[threadIdx.x - (blockDim.x - n)] = ((halo_index_left < 0) ? 0 : da[halo_index_left]);
    }

    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < n) { // Make the first n threads load the right halo cells
        sa[n + blockDim.x + threadIdx.x] = ((halo_index_right < width) ? da[halo_index_right] : 0);
    }

    sa[n + threadIdx.x] = da[tx];
    __syncthreads();

    int pvalue = 0;
    for (int j = 0; j < mask_width; ++j) {
        pvalue += sa[threadIdx.x + j] * dm[j];
    }
    dp[tx] = pvalue;
}

int main() {
    int *ha, *hm, *hp, *da, *dp;

    init(ha, N);
    init(hm, MASK_WIDTH);
    init(hp, N);
    cudaMemcpyToSymbol(dm, hm, MASK_WIDTH * sizeof(int)); // Transfer mask data to constant memory 

    cudaMalloc(&da, N * sizeof(int));
    cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dp, N * sizeof(int));

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
    cudaFree(dm);
    cudaFree(da);
    cudaFree(dp);

    return 0;
}