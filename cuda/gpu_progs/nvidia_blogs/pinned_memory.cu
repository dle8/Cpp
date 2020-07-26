Pinned Host Memory
Host (CPU) data allocations are pageable by default. The GPU cannot access data directly from pageable host memory, so when a data transfer from pageable host memory to device 
memory is invoked, the CUDA driver must first allocate a temporary page-locked, or “pinned”, host array, copy the host data to the pinned array, and then transfer the data from 
the pinned array to device memory. Pinned memory is used as a staging area for transfers from the device to the host. We can avoid the cost of the transfer between pageable 
and pinned host arrays by directly allocating our host arrays in pinned memory: cudaMallocHost() or cudaHostAlloc(), and deallocate it with cudaFreeHost().