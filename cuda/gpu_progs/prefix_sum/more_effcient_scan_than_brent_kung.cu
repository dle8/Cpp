/*
    We can design a parallel scan algorithm that achieves a higher work efficiency than does the Brent-Kung algorithm by adding a phase
    of fully independent scans on the subsections of the input. At the beginning of the algorihm, we partition the input section into 
    subsections. The number of subsections is the same as the number of threads in a thread block, one for each thread. During the first phase,
    each thread performs a scan on its subsection. 

    Notably, if each thread directly performs a scan by accessing the input from global memory, their accesses would not be coalesced. 
    For example, if we partition the input into multiple of section of 4, thread 0 would be access input element 0, thread 1 input element4.
    Therefore, one technique to be use is corner turning technique to improve coalescing. At the beginning of the phase, all threads collaborate
    to load the input into the shared memory iteratively. In each iteration, adjacent threads load adjacent elements to enable memory coalescing.

    Once all input elements are in the shared memory, the threads access their own subsection from the shared memory. 
    Step 1:
        - we do the scanning for each subsection. At the end of step 1, the last element of each section contains the sum of all input 
        elements in the section. 
    Step 2:
        - During the second phase, all threads in each block collaborately and perform a scan operation on a logical array that consists 
        of the last elements of all sections. This produce can be done with simple scan using Kogge-stone or Brent-Kung algorithm since 
        only a modest number (in this case number of threads in a block) of elements are involved. 
    Step 3:
        - Each thread adds to its element the new value of the last element of its predecessor's section. The last elements of each 
        subsection need not be update during this phase. 

    Using this three-phase approach we can use a much smaller number of threads than the number of elements in a section. The max size of the
    section is no longer limited by the number of threads in the block but rather, the size of the shared memory; all elements in the section
    must fit into the share memory.

    Advantage:
        - efficent use of execution resources. Assume using Kogge stone for phase 2. For an input list of N elements, if we use T threads,
        the amount of work done is N - 1 for phase 1, T * log2T for phase 2, N - T for phase 3. If we use P exection units, the execution
        takes (N - 1 + T*log2T + N - T) / P time units.
*/