/*
    One of the fastest parallel ways to produce sum values for a set of values is a reduction tree. With sufficient execution units, 
    a reduction tree can generate the sum for N values in log2N time units. The tree can also generate a number of subsums that can be used
    to calculate some scan output values.

    The idea is having two phases: reduction phase and post-reduction reverse phase.
    The reduction phase:
        - for iteration i, let the threads whose indexes take the form k * 2 ^ i - 1 to
        accumulate the results (*). This idea generates a number of sub sum that can be used to calculate
        some scan output values later. Total operations: (n / 2) + (n / 4) + ... + 1 = n - 1 operations
    The reverse phase:
        - distribute toe partial sums to the positions that can use these values as quickly as possible.
        stride value decreases from SECTION_SIZE/4 to 1 since SECTION_SIZE / 2 is already optimally calculated.
        - In each iteration, we need to push the value of the XY element from a position that is a multiple
        of the stride value - 1 to a position that is a stride away.
        - NUmber of operations is (2 - 1) + (4 - 1) + ... + (N / 4 - 1) + (N / 2 - 1), which is N - 1 - log2N.
        - SO the total scan is 2N - 2 - log2N.

    Advantage: 
        - As the input section increases, the algorithm never performs more than twice the number of operations performed by the sequential
        algorithm. 
    
    Disadvantage:
        - NUmber of active threads drops much faster through the reduction tree than simple scan. However, the inactive threads continue
        to consume execution resources in a CUDA device. Consequently, the amount of resources consumed is actually closer to
        (N / 2) * (2 * log2N - 1).

    Notice that: 
        - having more than SECTION_SIZE / 2 threads is unnecessary for reduction or distribution phase.
        - Therefore, we can launch a kernel with SECTION_SIZE / 2 threads in a block.

    Technique: 
        - TO avoid thread divergence, use a decreasing number of contiguous threads to perform the additions as the 
        loop advances (accrue the working threads to the front to avoid control divergence cause by alternative 
        differently-behaved threads) (**)

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

__global__ void brent_kung_scan(int *da, int* db, int sz) {
    __shared__ float smem[SECTION_SIZE];
    // multiple by two since the kernel takes care of 2 blocks at a time
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz) smem[threadIdx.x] = da[i];
    if (i + blockDim.x < sz) smem[threadIdx.x + blockDim.x] = da[i + blockDim.x];

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1; // (*)
        if (index < SECTION_SIZE) { // (**)
            smem[index] += smem[index - stride];
        }
    }

    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            smem[index + stride] += smem[index];
        }
    }
    
    for (int stride = SECTION_SIZE / 4; stride >= 4; stride / 2) {
        __syncthreads();
        
    }

    __syncthreads();
    if (i < sz) db[i] = smem[threadIdx.x];
    if (i + blockDim.x < sz) db[i + blockDim.x] = smem[threadIdx.x + blockDim.x];
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

    brent_kung_scan<<<grid, block>>>(da, db, N);
    cudaMemcpy(hb, db, N * sizeof(int), cudaMemcpyDeviceToHost);
    print(ha, N);
    print(hb, N);

    free(ha);
    free(hb);
    cudaFree(da);
    cudaFree(db);

    return 0;
}