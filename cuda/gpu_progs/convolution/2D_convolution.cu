/*
    If the width of the image in terms of bytes is not a multiple of the DRAm burst size, there is a misalignment from the DRAM burst
    boundaroes. Such misalignment can result in poor utilization of DRAM bandwidth when we attempt to access the data in one of the 
    rows. Therefore, padded elements to the end of each row such as each row ends at the DRAM burst boundaries.

    Pitch: the length of the rows after padding. 

    To calculate the linearized 1D index of the pixel elements, use pitch instead of width: idx = row * pitch + column (*). However, 
    when we iterate through a row, use width as the loop bound to ensure that we use only the original elements in a computation. (**)

    Benefit of 2D tiled kernel over basic kernel:
        - In basic kernel, each thread performs (MASK_WIDTH) ^ 2 accesses to the image array (load the 2D pixels of the image array).
        In total, each thread block performs (MASK_WIDTH) ^ 2 * (O_TILE_WIDTH) ^ 2 accesses.
        - In tiled kernel, all threads in a thread block collectively load one input file. Then total accesses for one block is
        (O_TILE_WIDTH + MASK_WIDTH - 1) ^ 2.

        => access ratio: (MASK_WIDTH) ^ 2 * (O_TILE_WIDTH) ^ 2 / (O_TILE_WIDTH + MASK_WIDTH - 1) ^ 2
*/

__global__ void convolution(float* p, float* n, int height, int width, int pitch, int channels, int mask_width, const float __restrict__ *M) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;

    int row_i = row_o - mask_width / 2;
    int col_i = col_o - mask_width / 2;

    __shared__ float n_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE + MASK_HEIGHT - 1];
    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < height)) {
        n_ds[ty][tx] = n[row_i * pitch + col_i]; // (*)
    } else {
        n_ds[ty][tx] = 0.0f;
    }

    float output = 0;
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
        for (int i = 0; i < MASK_WIDTH; ++i) {
            for (int j = 0; j < MASK_WIDTH; ++j) {
                output += M[i][j] * n_ds[i + ty][j + tx];
            }
        }
        if (row_o < height && col_o < width) {
            n[row_o * width + col_o] = output; // (**)
        }
    }
}