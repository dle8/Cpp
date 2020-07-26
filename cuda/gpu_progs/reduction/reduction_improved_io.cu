/*
    The local variable input has a substantial amount of load/store request. Such massive I/O impacts the thread blcock's scheduling due
    to the operational dependencu. The worst thing in the current data accumulation is that it has a dependency on device memory. So, we will
    use extra registers to issue more load instructinons to ease the dependency.

    This code uses 3 more registers to collect global memory data. The value of NUM_LOAD can vary depending on the GPU because it is affected
    by the GPU's memory bandwidth and the number of CUDA cores in a GPU
*/

#define NUM_LOAD 4
__global__ void reduction_kernel(float* g_out, float *g_in, unsigned int size) {
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_data[];

    // Maximizing memory bandwidth with grid-strided loops. Sequential threads load consecutive locations in global mem array g_in
    float input[NUM_LOAD] = 0.f;
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x) {
        input += g_in[i];
        for (int step = 0; step < NUM_LOAD; ++step) {
            input[step] += (i + step * blockDim.x * gridDim.x < size) ? g_in[i + step * blockDim.x * gridDim.x] : 0.f;
        }
    }
    for (int i = 1; i < NUM_LOAD; ++i) input[0] += input[i];
    s_data[threadIdx.x] = input[0];
    __syncthreads();

    // do reduction - sequential addressing to avoid thread divergence in reduction_shared_kernel.cu
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        g_out[blockIdx.x] = s_data[0];
    }
}