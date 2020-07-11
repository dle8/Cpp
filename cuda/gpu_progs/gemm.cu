#include <bits/stdc++.h>
#include <stdio.h>
using namespace std;

const int N = 2;
const int M = 3;
const int K = 4;
const int TILE_WIDTH = 16;

void print(int *m, int sz) {
    for (int i = 0; i < sz; ++i) {
        cout << m[i] << " ";
    }
    cout << '\n';
}

void init(int* &a, int sz) {
    a = (int*) malloc(sz * sizeof(int));
    srand (time(NULL));
    for (int i = 0; i < sz; ++i) {
        a[i] = rand() % 10;
    }
}

__global__ void gemm(int* da, int* db, int* dc, int n, int m, int k) {
    __shared__ int sa[TILE_WIDTH][TILE_WIDTH];
    __shared__ int sb[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    // bx = blockIdx.x, by = blockIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;


    int steps = (m - 1) / TILE_WIDTH + 1;
    int pvalue = 0;
    // printf("%d %d\n", row, col);
    for (int ph = 0; ph < steps; ++ph) {
        // sa[ty][tx] = da[row][ph * TILE_WIDTH + tx];
        int tmp = ph * TILE_WIDTH + tx;
        if (row < n && tmp < m) {
            sa[ty][tx] = da[row * m + tmp];
            // printf("sa %d %d %d\n", ty, tx, row * m + tmp);
        }
        
        // sb[ty][tx] = db[ph * TILE_WIDTH + ty][col];
        tmp = ph * TILE_WIDTH + ty;
        if (tmp < m && col < k) {
            sb[ty][tx] = db[tmp * k + col];
            // printf("sb %d %d %d\n", ty, tx, tmp * k + col);
        }
        
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; ++i) {
            pvalue += sa[ty][i] * sb[i][tx];
        }
        __syncthreads();
    }
    if (row < n && col < k) dc[row * k + col] = pvalue;
}

int main() {
    int *ha, *hb, *hc;
    init(ha, N * M);
    init(hb, M * K);
    init(hc, N * K);

    int *da, *db, *dc;
    cudaMalloc(&da, N * M * sizeof(int));
    cudaMalloc(&db, M * K * sizeof(int));
    cudaMalloc(&dc, N * K * sizeof(int));

    cudaMemcpy(da, ha, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, M * K * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N - 1) / TILE_WIDTH + 1, (K - 1) / TILE_WIDTH + 1);
    gemm<<<grid, block>>>(da, db, dc, N, M, K);

    cudaMemcpy(hc, dc, N * K * sizeof(int), cudaMemcpyDeviceToHost);

    print(ha, N * M);
    print(hb, M * K);
    print(hc, N * K);
    free(ha);
    free(hb);
    free(hc);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}