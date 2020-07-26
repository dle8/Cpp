/*
    To see the difference between global 
*/

#include <bits/stdc++.h>
using namespace std;

__global__ void coalesced_access_pattern(int *m, int dimx, int dimy, int sz, int mul) {
    // Calculate matrix cell coordinate based on thread indexes
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * dimx + col;

    // Check out bound condition
    for (; idx < sz; idx += blockDim.x * gridDim.x) {
        m[idx] *= mul;
    }
}

__global__ void uncoalesced_access_pattern(int *m, int dimx, int dimy, int sz, int mul) {
    // Calculate matrix cell coordinate based on thread indexes
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = col * dimx + row;

    // Check out bound condition
    for (; idx < sz; idx += blockDim.x * gridDim.x) {
        m[idx] *= mul;
    }
}

void print(int *m, int sz) {
    for (int i = 0; i < 10; ++i) cout << m[i] << " ";
    cout << '\n';
}

int main() {
    unsigned int n = 4 * 1024 * 1024, bytes = n * sizeof(int);

    int *d_matrix, *h_matrix;
    // A linearlized matrix with side = 2 * 1024
    h_matrix = (int*) malloc(bytes);
    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        h_matrix[i] = rand() % 10;
    }
    cudaMalloc(&d_matrix, bytes);
    cudaMemcpy(d_matrix, h_matrix, bytes, cudaMemcpyHostToDevice);

    // print(h_matrix, n);

    dim3 block(16, 16);
    dim3 grid((2 * 1024 - 1) / block.x + 1, (2 * 1024 - 1) / block.y + 1);
    coalesced_access_pattern<<<grid, block>>> (d_matrix, 2 * 1024, 2 * 1024, n, 3);
    // uncoalesced_access_pattern<<<grid, block>>> (d_matrix, 2 * 1024, 2 * 1024, n, 3);
    cudaMemcpy(h_matrix, d_matrix, bytes, cudaMemcpyDeviceToHost);

    // print(h_matrix, n);
    free(h_matrix);
    cudaFree(d_matrix);
    return 0;
}