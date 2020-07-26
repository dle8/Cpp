/*
    CSR: Compressed Sparse Row (CSR) stores only nonzero values in a one-dimensional data storage:
        - data[]: array to store data
        - col_index[]: array of column index of every nonzero value in the original sparse matrix
        - row_ptr[]: array of the starting location of every row in the compressed format as the size of each row varies after the
        zero elements are removed. In addition, the last element indicates the starting location of unexisted row to indicate where
        the last row last.

            for example:
            [
                3, 0, 1, 0
                0, 0, 0, 0
                0, 2, 4, 1
                1, 0, 0, 1
            ]

            CSR form:
            data[7]      = {3, 1, 2, 4, 1, 1, 1}
            col_index[7] = {0, 2, 1, 2, 3, 0, 3}
            row_ptr[5]   = {0, 2, 2, 5, 7}
        
        CSR removes all zero elements from the storage. Even though it incurs more storage due to col_index[] and row_ptr[], but
        number of zero elements outweighs these two array in large sparse matrix. Removing all zero elements from the storage also
        eliminates the need to fetch these zero elements from memory or to perform useless multiplication ops.

        Problem:
            - Kernel does not make coalesced memory accesses (*)
            - control flow divergence in all warps as number of iterations performed by a thread in the dot product loop depends on
            the number of nonzero elements in the row assigned to the thread. Since the distribution of non zero element among
            rows can be random, adjacent rows can have varying numbers of non zero elements. (**)

*/

/* A sequential loop that implements SpMV (Sparse Matrix-Vector) multiplication on the CSR format

    for (int row = 0; row < num_rows; ++row) {
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int elem = row_start; elem < row_end; ++elem) {
            dot += data[elem] * x[col_index[elem]];
        }
        y[row] += dot;
    }
*/

__global__ void SpMV_CSR(int num_rows, float* data, int* col_index, int *row_ptr, float* x, float *y) {
    // Each thread handles one row
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < num_rows) {
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int elem = row_start; elem < row_end; ++elem) { // (**)
            dot += data[elem] * x[col_index[elem]]; // (*)
        }
        y[row] = dot;
    }
}