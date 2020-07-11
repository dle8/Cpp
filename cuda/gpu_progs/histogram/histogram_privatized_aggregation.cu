/*
    Still using the idea of privatization to avoid high traffic for high contended output data, but used for datasets that have a
    large concentration of identical data values in localized areas. (eg pictures of the sky can have large patches of pixels of
    identical value).

    Optimization: each thread aggregate consecutive updates into a single update if they are updating the same element of histogram.
    Such aggregation redices the number of atomic operations to the highly contended histogram elements, thus improving the effective
    throughput of the computation.
*/

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

__global__ void histogram_privatization(char* buffer, int sz, int* bins, int num_bins) {
    extern __shared__ int sbins[];
    for (int binidx = threadIdx.x; binidx < num_bins; binidx += blockDim.x) {
        sbins[binidx] = 0;
    }
    __syncthreads();

    int prev_index = -1, accumulator = 0, idx = threadIdx.x + blockDim.x * blockIdx.x, stride = blockDim.x * gridDim.x;
    for (int i = idx; i < sz; i += stride) {
        int alphabet_pos = buffer[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26) {
            int curr_index = alphabet_pos / 4;
            if (curr_index != prev_index) {
                if (prev_index >= 0) atomicAdd(&sbins[prev_index], accumulator);
                accumulator = 1;
                prev_index = curr_index;
            }
            else {
                ++accumulator;
            }
        }
    }
    if (accumulator > 0) atomicAdd(&sbins[prev_index], accumulator);

    __syncthreads();
    // Write back values from shared memory bins to global memory bins
    for (int binidx = threadIdx.x; binidx < num_bins; binidx += blockDim.x) {
        atomicAdd(&bins[binidx], sbins[binidx]);
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

    histogram_privatization<<<grid, block, NUM_BINS * sizeof(int)>>>(dstr, hstr.size(), dbins, NUM_BINS);

    cudaMemcpy(hbins, dbins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    print(hbins, NUM_BINS);

    free(hbins);
    cudaFree(dbins);
    return 0;
}