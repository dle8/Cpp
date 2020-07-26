/*
    Optimization:
    - Instead of reading from global and writing back to global after each stride iteration, only load from global for
    the first stride, use shared memory as storage for intermediate values. Read values from the previous stride stored in
    shared memory, and write the results of the current stride to shared memory again. Use __syncthreads() to make sure
    every thread finishes it exeuction in the current iteration before moving onto the new iteration. (*)
*/


#include <stdio.h>
#include "reduction.h"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

// cuda thread synchronization
__global__ void
reduction_kernel(float* d_out, float* d_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;

    __syncthreads();

    // do reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // thread synchronous reduction
        // if ( (idx_x % (stride * 2)) == 0 ) // causing warp divergence
        if (idx_x & (stride * 2 - 1) == 0) // causing warp divergence
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads(); // (*)
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}

void reduction(float *d_out, float *d_in, int n_threads, int size)
{   
    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);
    while(size > 1)
    {
        int n_blocks = (size + n_threads - 1) / n_threads;
        reduction_kernel<<< n_blocks, n_threads, n_threads * sizeof(float), 0 >>>(d_out, d_out, size);
        size = n_blocks;
    } 
}