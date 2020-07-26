CUDA Streams
A stream in CUDA is a sequence of operations that execute on the device in the order in which they are issued by the host code. Operations within a stream are guaranteed to execute in the prescribed order while 
operations in different streams can be interleaved and, when possible, they can even run concurrently. CUDA Applications manage concurrency by executing asynchronous commands in streams, 
sequences of commands that execute in order.

<<<<<<<<<<<<<<<<The default stream >>>>>>>>>>>>>>
All device operations (kernels and data transfers) in CUDA run in a stream. When no stream is specified, the default stream (also called the “null stream”) is used. The default stream is different from other streams because it is a synchronizing stream with 
respect to operations on the device: no operation in the default stream will begin until all previously issued operations in any stream on the device have completed, and an operation in the default stream must complete before any other operation (in any 
stream on the device) will begin.

New feature: use a separate default stream per host thread, and to treat per-thread default streams as regular streams (i.e. they don’t synchronize with operations in other streams)
Let’s look at some simple code examples that use the default stream, and discuss how operations progress from the perspective of the host as well as the device.

/*
    cudaMemcpy(d_a, a, numBytes, cudaMemcpyHostToDevice);
    increment<<<1,N>>>(d_a)
    cudaMemcpy(a, d_a, numBytes, cudaMemcpyDeviceToHost);
*/

The asynchronous behavior of kernel launches from the host’s perspective makes overlapping device and host computation very simple. 
We can modify the code to add some independent CPU computation as follows.

/*
    cudaMemcpy(d_a, a, numBytes, cudaMemcpyHostToDevice);
    increment<<<1,N>>>(d_a)
    myCpuFunction(b)
    cudaMemcpy(a, d_a, numBytes, cudaMemcpyDeviceToHost);
*/

<<<<<<<<<<<<<<<<Non-default streams >>>>>>>>>>>>>>
Non-default streams in CUDA C/C++ are declared, created, and destroyed in host code as follows.
/*
    cudaStream_t stream1;
    cudaError_t result;
    result = cudaStreamCreate(&stream1)
    result = cudaStreamDestroy(stream1)
*/
To issue a data transfer to a non-default stream we use the cudaMemcpyAsync() function, which is similar to the cudaMemcpy() function discussed in the previous post, but takes a stream identifier as a fifth argument.

// result = cudaMemcpyAsync(d_a, a, N, cudaMemcpyHostToDevice, stream1)
cudaMemcpyAsync() is non-blocking on the host, so control returns to the host thread immediately after the transfer is issued. 
Variant for 2D and 3D array: cudaMemcpy2DAsync(), cudaMemcpy3DAsync()

To issue a kernel to a non-default stream1:

// increment<<<1,N,0,stream1>>>(d_a)

<<<<<<<<<<<<<<<<Synchronization with streams >>>>>>>>>>>>>>
Synchronize the host code with operations in a stream
    - cudaDeviceSynchronize(): blocks the host code until all previously issued operations on the device have completed. (Overkill, can hurt performance)
    - cudaStreamSynchronize(stream): block the host thread until all previously issued operations in the specified stream have completed
        - cudaStreamQuery(stream): tests whether all operations issued to the specified stream have completed, without blocking host execution
        - cudaStreamWaitEvent(event): synchronize operations within a single stream on a specific event (event may be recorded in different stream or different device)
        - cudaEventSynchronize(event) + cudaEventQuery(event) act similar to their stream counterparts, except that their result is based on whether a specified event has been recorded rather than whether a specified stream is idle

    
<<<<<<<<<<<<<<<< Overlapping Kernel Execution and Data Transfers >>>>>>>>>>>>>>
Earlier we demonstrated how to overlap kernel execution in the default stream with execution of code on the host. But our main goal in this post is to show you how to overlap kernel execution with data transfers. There are several requirements for this to happen.

The device must be capable of “concurrent copy and execution”.  
The kernel execution and the data transfer to be overlapped must both occur in different, non-default streams.
The host memory involved in the data transfer must be pinned memory.

In the modified code, we break up the array of size N into chunks of streamSize elements. Since the kernel operates independently on all elements, each of the chunks can be processed independently. 
The number of (non-default) streams used is nStreams=N/streamSize. 
Case 1: loop over all the operations for each chunk of the array:

/*
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
        cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }
*/
Case 2: batch similar operations together, issuing all the host-to-device transfers first, then kernel launches, then device-to-host transfers:

/*

    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, cudaMemcpyHostToDevice, stream[i]);
    }

    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    }

    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&a[offset], &d_a[offset], treamBytes, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToHost, stream[i]);
    }
*/
