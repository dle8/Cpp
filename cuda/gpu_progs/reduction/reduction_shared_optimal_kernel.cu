__global__ void sequential_reduction_kernel(float* g_out, float *g_in, unsigned int size) {
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_data[];
    // Maximizing memory bandwidth with grid-strided loops. Sequential threads load consecutive locations in global mem array g_in
    float input = 0.f;
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x) input += g_in[i];
    s_data[threadIdx.x] = input;
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