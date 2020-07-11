#include <bits/stdc++.h>
using namespace std;

const int NUM_BINS = 7;
const int NUM_BLOCKS = 16;

void init(int* &a, int sz) {
    a = (int*) calloc(sz, sizeof(int));
}

void print(int* a, int sz) {
    for (int i = 0; i < sz; ++i) {
        cout << a[i] << " ";
    }
    cout << '\n';
}

__global__ void histogram(char* buffer, int sz, int* bins) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // All threads handle blockDim.x * gridDim.x consecutive elements in each iteration
    for (int i = idx; i < sz; i += blockDim.x * gridDim.x) {
        int alphabet_pos = buffer[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26) atomicAdd(&bins[alphabet_pos / 4], 1);
    }
}

int main() {
    string hstr;
    cin >> hstr;
    int * hbins, *dbins;
    char* dstr;
    init(hbins, NUM_BINS);
    cudaMalloc(&dbins, NUM_BINS * sizeof(int));
    cudaMalloc(&dstr, hstr.size() * sizeof(char));
    cudaMemcpy(dstr, hstr.c_str(), hstr.size() * sizeof(char), cudaMemcpyHostToDevice);

    dim3 block(NUM_BLOCKS);
    dim3 grid((hstr.size() - 1) / NUM_BLOCKS + 1);

    histogram<<<grid, block>>>(dstr, hstr.size(), dbins);

    cudaMemcpy(hbins, dbins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    print(hbins, NUM_BINS);

    free(hbins);
    cudaFree(dbins);
    return 0;
}