/*
    A parallel merge sort divides the output array (eg. array C) into multiple sections and derived the two portions of two input 
    arrays (eg. array A and B) that needed to be merged in order to generate the chosen output sections. (assume that A element
    will stand before B element if there's a tie)

    Steps:
        - Each thread calculates the range of output positions (output range) that it is going to produce, and uses that range as
        the input to a CO-RANK FUNCTION (a function that identifies the corresponding input ranges that will be merged to produce
        chosen output sections). In other words, the range of input elements to be used by each thread is a function of the input
        values

        - Once the input and output ranges are determined, each thread can independently access its two input subarrays and sequential
        merge them into the output array.
    
    Observation:
        - For any k such that 0 <= k < m + n, we can find i and j such that k = i + j, 0 <= i < m and 0 <= j < n and the subarray
        C[0] - C[k - 1] is the result of merging subarray A[0] - A[i - 1] and subarray B[0] - B[j - 1]

        - k: rank, i, j: co-rank (i and j is unique)

    Therefore, the input subarrays to be used by thread t are defined by the co-rank values for thread (t - 1) and thread t:
        A[i_(t- 1)] to A[i_t - 1] and B[j_(t - 1)] to B[j_t - 1].

*/

#include <bits/stdc++.h>
using namespace std;

const int BLOCK_SIZE = 16;
const int NUM_SIZE = 10;

// Take the rank(k) of C array and information about the two input arrays and return i co-rank value
// O(log n)
int co_rank(int k, int *A, int m, int *B, int n) {
    int i = k < m ? k : m; // i = min(k, m)
    int j = k - i;
    // i_low and j_low are smallest possible co-rank values that could be generated by this function
    int i_low = 0 > (k - n) ? 0 : k - n; // i_low = max(0, k - n)
    int j_low = 0 > (k - m) ? 0 : k - m; // i_low = max(0, k - m)
    int delta;
    // We want to find i and j such that A[i - 1] <= B[j] (largest A element in current subarray can 
    // be equal to the smallest B elements in the next subarray) and B[j - 1] < A[i]
    while (true) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = ((i - i_low + 1) >> 1); // ceil((i - i_low) / 2)
            j_low = j;
            j += delta;
            i -= delta;
        }
        else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i += delta;
            j -= delta;
        } else break;
    }
    return i;
}

__global__ void merge_basic_kernel(int * A, int *B, int *C, int m, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int section_size = (m + n - 1) / (blockDim.x * gridDim.x) + 1;
    int k_curr = tid * section_size;
    int k_next = min((tid + 1) * section_size, m + n);
    int i_curr = co_rank(k_curr, A, m, B, n), j_curr = k_curr - i_curr;
    int i_next = co_rank(k_next, A, m, B, n), j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

void merge_sort(int *arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main() {
    int *da, *ha;
    ha = (int*) malloc(NUM_SIZE * sizeof(int));
    for (int i = 0; i < NUM_SIZE; ++i) ha[i] = rand() % 10;
    cudaMalloc(da, NUM_SIZE * sizeof(int));
    cudaMemcpy(da, ha, NUM_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_SIZE - 1) / BLOCK_SIZE + 1);

    merge_basic_kernel<<<grid, block>>>()

    cudaMemcpy(ha, da, NUM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(da);
    free(ha);
    return 0;
}