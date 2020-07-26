/*
    The three kernel presented thus far assume that the entire input can be loaded in the shared memory. Obviously, we cannot expect all input
    elements fit into shared mem. 

    Hierarchical approach:
        - Partition the input into sections so that each of them can fit into the shared memory and be pricessed by a single block. For Pascal
        that I am currently using, the Brent Kung kernel can process up to 2048 elements in each section by using 1024 threads in each blocks.
        (Optional - consider if this should be said: With max 65536 thread blocks max in x-dimension, this process can process up to ~ 100
        millions elements).
    
    Assume running one of the three kernels in large input dataset. At the end of the grid execution, the Y array contains the scan results for 
    individual sections, called scan blocks. Each result value in a scn block only contains the accumulated values of all preceding elements
    within the same scan block. These scans blocks need to be combined into the final result, which means we need to write and launch another 
    kernel that adds the sum of all elements in precedding scan blocks to each element of a scan block. 

    We can iuse Kogge -Stone kernel, the Brent_kung kernel, or the 3 phase kernel to process the individual scan blocks. The last element of each
    scan block yields the sum (depends on the operations) of all input elements of the scan block. After that, we gather the last result elements
    from each scan block into an array, for example, S and performs a scan on these output elements. This can be a carried out by changing the 
    code at the end of the scan kernel so that the last thread of each block writes its result to an S array by using its blockIdx.x as index. 
    A scan operation is then performed on S to produce the second-level scan output values, which are the accumulated sum from the starting location
    X[0] to the end of each scan block. The second-level scan output values are added to the values of their corresponding scan blocks.

    The hierarchical scan can be implemented with 3 kernels:
    Kernel 1 is largely the same as the 3-phase kernel. We need to add a parameter S, which has the dimension of InputSize/Section_SIZE. At the
    end of the kernel, we add a conditional statement. The last thread in the block writes the output value of the last XY element in the scan
    block to the blockIdx.x position of S:
            
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) S[blockIdx.x] = XY[SECTION_SIZE - 1];
    
    The second kernel is simple one of the 3 parallel scan kernels, which takes S as input and write S as output.

    The third kernel takes the S and Y arrays as inputs and writes its output back into Y. Assuming that we launch the kernel with SECTION_SIZE
    threads in each block, each thread adds one of the S elements (selected by blockIdx.x - 1) to one Y elements:
            
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        Y[i] += S[blockIdx.x - 1];
    
    The threads in a block add the sum of the previous scan block to the elements of their scan block.
*/