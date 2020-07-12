/*
    Hybrid approach to regulate padding in ELL
    The Coordinate (COO) format:
    - a regularization technique to curb the length of longest non-zero rows in the CSR format or the ELL format to reduce excessive 
    padding and data fetching.
    - have two more arrays outside of data:
        - col_index[i] and row_index[i] tells in what column and row in the original matrix that the data belongs to
        - So for each element data[i], we can simply perform y[row_index[i]] += data[i] * x[col_index[i]]
    
    Steps:
        1. Before converting a sparse matrix from the CSR format to ELL format, remove some elements from rows with exceddingly large
        numbers of non-zero elements and place them into a separate COO storage -> reduce extra-long rows
        2. Use SpMV/ELL on the remaining elements.
        3. Use sequential/ parallel SpMV/ COO to add the missing contributions from the elements in the COO representation.
    
    Does it worth it?
        - In situations where SpMV is performed on the same sparse kernel repeatedly in an iterative solver (like Jacobi), the x and
        y vectors vary, but the sparse matrix remains the same because its element correspond to the coefficients of the linear system
        of equations begin solved

        => Work done to produce both hybrid ELL and COO representations can be amortized across many iterations.
*/

// Sequential SpVM/COO
/*
    for (int i = 0; i < num_elem; ++i) {
        y[row_index[i]] += data[i] * x[col_index[i]];
    }
*/

// Parallel SpVM/COO
__global__ void spvm_coo(float* data, float* y, float* x, int* row_index, int* col_index, int sz) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int section_size = (sz - 1) / (blockDim.x * gridDim.x) + 1;
    int start = tid * section_size;

    for (int k = 0; k < section_size; ++k) {
        for (int i = start + k; i < sz; ++i) {
            atomicAdd(&y[row_index[i]], data[i] * x[col_index[i]]); // Multiple threads can access the same position in y vector
        }
    }
}