/*
    Parallel BFS assigns each section of the previous frontier array to each thread block. The threads divide the section in
    an interleaved manner to enable coalesced memory access to the p_frontier array. (*)

    For each previous frontier vertex, a thread will likely write multiple vertices into the c_frontier array. This creates a global
    uncoalesced memory write pattern. Use privatized bufeer in the shared memory to assemble the contribution by the threads
    in a block (**), and have threads to write the contents of the shared memory buffer into the global memory in a coalesced manner at
    the end of the kernel (***). This privatized buffer is called BLOCK-LEVEL QUEUE. Also need to create a privatized c_frontier_tail_s
    variable in the shared memory for insertion into the block level queue.

    Optimizations:
        - Memory bandwidth: Since access to the edges, dest, and label array are not coalesced in general, accesses to these arrays should
        go through the texture memory
        - even block level queue can suffer heavy contention. Use a finer-grain warp queues by classifying threads into the same number of
        classes as the number of warp-level queues using the lease significant bits of their threadIdx.x values. If warp queue overflows, write 
        to block-level queue. If block-level queue overflows, write to global memory instead
        - For most graphs, the frontier of the first several iterations of a BFS can be quite small, so kernel launch overhead may outweigh the
        benefit of parallelism -> use another kernel with one thread block that uses only a block queue. If the number of vertices grows to the
        extent such that block level queue overflows, the kernel copies the block level queues to global & return to the host code, then call
        the regular kernel.
*/

void BFS_host(unsigned int source, unsigned int* edges, unsigned int *dest, unsigned int* label) {
    // allocate edges_d, dest_d, label_d, and visited_d in device global memory
    // copy edges, dest, and label to device global memory
    // allocate frontier_d, c_frontier_tail_d, p_frontier_tail_d in device global memory

    unsigned int *c_frontier_tail_d = &frontier_d[0];
    unsigned int *p_frontier_tail_d = &frontier_d[MAX_FRONTIER_SIZE];

    // launch a simple kernel to init:
    // all visited_d elements to 0 except source to 1
    // *c_frontier_tail_d = 0;
    // p_frontier_tail_d[0] = source;
    // *p_frontier_tail_d = 1;
    // label[source] = 0;

    p_frontier_tail = 1;
    while (p_frontier_tail > 0) {
        int num_blocks = (p_frontier_tail - 1) / BLOCK_SIZE + 1;
        BFS_BQueue_kernel<<<num_blocks, BLOCK_SIZE>>>(\
            p_frontier_tail, p_frontier_tail_d, c_frontier_tail, c_frontier_tail_d, edges_d, dest_d, label_d, visited_d \
        );
        // use cudaMemcpy to read the *c_frontier_tail value back to host and assign it to p_frontier_tail for the while-loop
        // condition test

        int* temp = c_frontier_d; c_frontier_d = p_frontier_d; p_frontier_d = temp; // swap the roles
        // launch a simple kernel to set *p_frontier_tail_d = *c_frontier_tail_d; *c_frontier_tail_d = 0;
    }
}

__global__ void BFS_BQueue_kernel(unsigned int *p_frontier, unsigned int *p_frontier_tail, unsigned int *c_frontier, \
    unsigned int*c_frontier_tail, unsigned int* edges, unsigned int *dest, unsigned int * label, unsigned int* visited) {
    __shared__ unsigned int c_frontier_s[BLOCK_QUEUE_SIZE]; // (**)
    __shared__ unsigned int c_frontier_tail_s, our_c_frontier_tail;

    if (threadIdx.x == 0) c_frontier_tail_s = 0;
    __syncthreads();
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *p_frontier_tail) {
        const unsigned int my_vertex = p_frontier[tid];  // (*)
        for (unsigned int i = edges[my_vertex]; i < edges[my_vertex + 1]; ++i) {
            const unsigned int was_visited = atomicExch(&visited[dest[i]], 1); // check if this vertex was visited
            if (!was_visited) {
                label[dest[i]] = label[my_vertex] + 1;
                const unsigned int my_tail = atomicAdd(&c_frontier_tail_s, 1); // add into privatized block queue buffer
                if (my_tail < BLOCK_QUEUE_SIZE) {
                    c_frontier_s[my_tail] = dest[i]; // (**)
                } else { // block-level queue overflows, so stored into the c_frontier directly
                    c_frontier_tail_s = BLOCK_QUEUE_SIZE;
                    const unsigned int my_global_tail = atomicAdd(c_frontier_tail, 1);
                    c_frontier[my_global_tail] = dest[i];
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // reverse a section to write this block privatized buffer into global memory. Subsequent blocks will write starting from current c_frontier_tail + c_frontier_tail_s
        our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s); 
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < c_frontier_tail_s; i += blockDim.x) { // coalesced write to the global c_frontier array.
        c_frontier[our_c_frontier_tail + 1] = c_frontier_s[i]; // (***)
    }
}