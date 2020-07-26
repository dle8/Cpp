/*
    Kogge -stone algorithm

    The main objective is to create each element quickly by calculating a reduction tree of the relevant input elements for each output
    element. Before the algorithm begins, assume XY[i] contains the input elment x[i]. At the end of iteration n, XY[i] will contain
    the sum of up to 2^n input elements at and before the location.

    We assign each thread to evolve the contents of one XY element. We write kernel that performs scan on 1 section of the input that is
    small enought for a block to handle. The size of a section is defined as the compile-time constant SECTION_SIZE. We assume that the
    kernel launch will use BLOCK_SIZE as the block size so that the number of threads is equal to the number of section elements. Each thread
    will be responsible for calculating one output elements.

    The loop iterates through the reduction tree for the XY array position assigned to a thread. Use __syncthread to ensure all threads have
    finished their previous iteration of additions in the reduction tree before any of them starts the next iteration.

    Speed and efficiency:
        - All threads iterates up to Log2N steps, N = section_size. In each iteration, the number of inactive threads = stride size. Therefore,
        amount of work done for the algorithms is: sigma(N - stride), for stride = 1, 2, 4, .., N/2 (so Log2N terms) = N*log2N - (N - 1). So
        the kernel do more work and the sequential algorithm.

        - Assume the sequential scan tkaes N time units to process N input elements. With P exeuction unit (SM), we can expect the kernel
        to execute for (N * log2N) / P time units. 

    Advantage:
        - Good execution spped given sufficient hardware resource.
        - This kernel is typically used to calculate the scan result for a section with a modest number of elements, such as 32, or 64 as its
        execution has very limited amount of control divergence. In newer GPU ar, its computation can be efficiently performed with shuffle
        instructions within warps.

    Disadvantages:
        - The use of hardware (SM) for executing the parallel kernel is much less efficient. 
        - The extra work consume additional enery. This additional demand makes the kernel less appropriate for power-constrained envs such as
        mobile applications.

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
    __shared__ int XY[SECTION_SIZE];
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Inclusive scan
    if (i < sz) XY[threadIdx.x] = da[i];

    // Exclusive scan
    /*
    if (i < sz && threadIdx.x != 0) {
        XY[threadIdx.x] = da[i - 1];
    } else XY[threadIdx.x] = 0;
    */
    
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (threadIdx.x >= stride) XY[threadIdx.x] += XY[threadIdx.x - stride];
    }

    db[i] = XY[threadIdx.x];
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