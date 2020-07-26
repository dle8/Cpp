/*
    Tiled 1D convolution kernel using constant memory. Constant memory used to contain the convolution mask (M array in this program). 
    Since they are not chagned during the kernel execution, the hardware can aggressively cache the constant variable values in L1 (because 
    if they don't change then there is no need to have cache coherence mechanism). Furthermore, the design of caches is typically optimized 
    to broadcast a value to a large number of threads. As a result, when all warp access the same constant memory varible, as in the case of 
    convolution masks, the caches can provide tremendous amount of bandwidth to satisfy the data needs of threads. Also, since the size of 
    the masks is typically small, we can assume that all mask elements are effectively always accessed from caches.

    Notation:
        - The elemnts that are involved in multiple tiles and loaded by multiple blocks are called halo cells since the "hang" from the side
        of the part that is used solely by a single block.
        - The center part of an input tile that is used solely by a single block are the internal cells of that input file.

    Cost and benefit:
        - O_TILE_WIDTH+MASK_WIDTH -1 elements loaded for each input tile
        – O_TILE_WIDTH*MASK_WIDTH global memory accesses replaced by shared memory accesses
        - reduction factor of (O_TILE_WIDTH * MASK_WIDTH) (each thread using global memory to read MASK_WIDTH elements of mask M) / (O_TILE_WIDTH + MASK_WIDTH - 1)
*/

#include <bits/stdc++.h>
using namespace std;

const int N = 7;
const int MASK_WIDTH = 5;
const int BLOCK_SIZE = 16;

__constant__ int M[MASK_WIDTH];

__global__ void convolution(int* N, int* P, int mask_width, int width) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int N_ds[N + MASK_WIDTH / 2];

    int n = mask_width / 2;

    // Make the last threads load the first hallo cells and make the first threads load the last hallo cells. This helps
    // reduce number of global memory loads -> each element needed in the input for the convolution is loaded once. In this
    // design, ome threads need to load more than one input element into the shared memory, and all threads participate 
    // in calculating output elements
    
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x > blockDim.x - n) { // Make the last n threads load the left hallo cells
        N_ds[threadIdx.x - (blockDim.x - n)] = ((halo_index_left < 0) ? 0 : N[halo_index_left]);
    }

    N_ds[n + threadIdx.x] = N[i];

    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < n) { // Make the first n threads load the right halo cells
        N_ds[n + blockDim.x + threadIdx.x] = ((halo_index_right < width) ? N[halo_index_right] : 0);
    }
    __syncthreads();

    int pvalue = 0;
    for (int j = 0; j < mask_width; ++j) {
        pvalue += N_ds[threadIdx.x + j] * M[j];
    }
    P[i] = pvalue;
}

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

int main() {
    int *ha, *hm, *hp, *da, *dp;

    init(ha, N);
    init(hm, MASK_WIDTH);
    init(hp, N);
    cudaMemcpyToSymbol(M, hm, MASK_WIDTH * sizeof(int)); // Transfer mask data to constant memory 

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
    cudaFree(M);
    cudaFree(da);
    cudaFree(dp);

    return 0;
}