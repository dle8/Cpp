#include <bits/stdc++.h>
using namespace std;

__global__ void transposeCoalesced(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
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