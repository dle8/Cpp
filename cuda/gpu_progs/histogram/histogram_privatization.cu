/*
    Privatization technique: replicate highly contended output data structures to private copies so that each thread or subsets of
    thread can update its private copy. 
    
    Benefit: 
        - private copies can be accessed with much less contention and often at much lower latency. 
        - private copies increase throughput for updating the data structure.

    Downside:
        - private copies need to be merged into the original data structure after the computation completes.
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

    int idx = threadIdx.x + blockDim.x * blockIdx.x, stride = blockDim.x * gridDim.x;
    for (int i = idx; i < sz; i += stride) {
        int alphabet_pos = buffer[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26) {
            // Update privatized shared memory bins
            atomicAdd(&sbins[alphabet_pos / 4], 1);
        }
    }

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