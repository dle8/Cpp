#include <bits/stdc++.h>
using namespace std;

__global__  void transpose_per_element_tiled(float* t, const float* m, int matrixSize) { 
   int col = blockIdx.x * blockDim.x + threadIdx.x;        // col 
   int row = blockIdx.y * blockDim.y + threadIdx.y;        // row 

   if (col >= matrixSize || row >= matrixSize) return; 

   extern __shared__ float tile[]; 

   // Coalesced read from global memory - TRANSPOSED write into shared memory 
   int from = row * matrixSize + col; 
   int ty   = threadIdx.y * blockDim.y + threadIdx.x;    // row 

   tile[ty] = m[from]; 
   __syncthreads(); 

   int tx   = threadIdx.y + threadIdx.x * blockDim.x;    // col 

   // Read from shared memory - coalesced write to global memory 
   int to   = (blockIdx.y * blockDim.y + threadIdx.x) + (blockIdx.x * blockDim.x + threadIdx.y) * matrixSize; 

   t[to] = tile[tx];
}

void print(float* a, int sz) {
    
}

int main() {
    float* hodata, *hidata, *dodata, *didata;
    int n;
    cin >> n;
    hidata = (float*) malloc(n * n * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < n * n; ++i) hidata[i] = rand() % 10;
    cudaMalloc(didata, n * n * sizeof(int));
    cudaMemcpy(didata, hidata, n * n * sizeof(int), cudaMemcpyHostToDevice);

    hodata = (float*) malloc(n * n * sizeof(float));
    cudaMalloc(dodata, n * n * sizeof(int));

    cudaFree(odata);
    free(idata);
    return 0;
}