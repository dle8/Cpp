/*
    Inclusive scan (cut points): takes (x[0], x[1], ..., x[n - 1]) into (x[0], x[0] ⊕ x[1], x[0] ⊕ x[1] ⊕ x[2], ..., x[0] ⊕ x[1] ... ⊕  x[n - 1])
    Exclusive scan (beginning points): takes (x[0], x[1], ..., x[n - 1]) into (0, x[0], x[0] ⊕ x[1] ⊕ x[2], ..., x[0] ⊕ x[1] ... ⊕  x[n - 2])

    In order to turn inclusive scan to exclusive scan, shift the elements to the right 1 pos, and fill the first element with 0.
    Vice versa, in order to turn exclusive scan to inclusive scan, shift the elements to the left 1 pos, and fill in the last element.

    In this example, the kernel performs scan on one section of the input that is small enough for a block to handle. The size of a
    section is defined as the compile-time constant SECTION_SIZE.
*/

#include <bits/stdc++.h>
using namespace std;

const int SECTION_SIZE = 16;
const int N = 10;

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

__global__ void scan(int *da, int* db, int sz) {
    __shared__ int smem[SECTION_SIZE];
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Inclusive scan
    if (i < sz) smem[threadIdx.x] = da[i];

    // Exclusive scan
    /*
    if (i < sz && threadIdx.x != 0) {
        smem[threadIdx.x] = da[i - 1];
    } else smem[threadIdx.x] = 0;
    */
    
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (threadIdx.x >= stride) smem[threadIdx.x] += smem[threadIdx.x - stride];
    }

    db[i] = smem[threadIdx.x];
}

int main() {
    int *ha, *hb, *da, *db;
    init(ha, N);
    init(hb, N);
    cudaMalloc(&da, N * sizeof(int));
    cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&db, N * sizeof(int));

    dim3 block(SECTION_SIZE);
    dim3 grid(1);

    scan<<<grid, block>>>(da, db, N);
    cudaMemcpy(hb, db, N * sizeof(int), cudaMemcpyDeviceToHost);
    print(ha, N);
    print(hb, N);

    free(ha);
    free(hb);
    cudaFree(da);
    cudaFree(db);

    return 0;
}