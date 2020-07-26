/*
    ELL storage format solves non-coalsced memory accesses and control divergence by padding and transposition on sparse matrix data.

    Padding:
        - Determine the rows with maximal number of non zero elements
        - Add dummy zeros to other rows after non elements so all rows have the same length -> no wrap divergence since all rows have
        same length
    
    Tranposition:
        - Lay matrix out in column major order, so all elements accessed by the threads are in consecutive memory locations.

    
        For example:
            [                               [                               [
                3, 0, 1, 0                      3, 1, *                         3, *, 2, 1
                0, 0, 0, 0                      *, *, *                         1, *, 4, 1
                0, 2, 4, 1                      2, 4, 1                         *, *, 1, *
                1, 0, 0, 1                      1, 1, *
            ]                               ]                               ]
                                            <CSR with padding>                  <Transposed>

            data[12]      = {3, *, 2, 1, 1, *, 4, 1, *, *, 1, *}
            col_index[7]  = {0, *, 1, 0, 2, *, 2, 3, *, *, 3, *}
    
        - col_index[0] to col_index[3] contains the column positions of the 0th elements of all rows.
        - row_ptr is no longer needed since the beginning of row i is simplified to data[i].
        - to move from the current element of row i to the next element: i + num_rows, with num_rows is the number of rows
        of the original matrix (*)

    Downside:
        - Screwed if there is a row that is exceptionally long -> more data to fetch and padding will extend the runtime
        of all warps of the SpMV/ELL kernel.
*/

// num_elem: maximal number of non zero elements among all rows in the original sparse matrix
__global__ void SpMV_ELL(int num_rows, float* data, int *col_index, int num_elem, float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        for (int i = 0; i < num_elem; ++i) {
            dot += data[row + i * num_rows] * x[col_index[row + i * num_rows]]; // (*)
        }
        y[row] += dot;
    }
}