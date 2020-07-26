#include <stdio.h>


__global__ void kernel_A( float *g_data, int dimx, int dimy, int niterations)
{
	//change global memory access so that we have coalesced access

	int ix = threadIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	int index = blockIdx.x*blockDim.x + ix;
	int idx = iy*dimx + index;

	float value = g_data[idx];
	
	if(ix & 1){

	    for(int i=0; i<niterations; i++)
	    {
	    	value += sqrtf( logf(value) + 1.f );
		}
	}

	else{

		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
	}

	g_data[idx] = value;
}

float timing_experiment( void (*kernel)( float*, int,int,int), float *d_data, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky);
	dim3 grid( dimx/block.x, dimy/block.y );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, dimx,dimy, niterations);
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

int main()
{
	int dimx = 2*1024;
	int dimy = 2*1024;

	int nreps = 10;
	int niterations = 20;

	int nbytes = dimx*dimy*sizeof(float);

	float *d_data=0, *h_data=0;
	cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
	printf("allocated %.2f MB on GPU\n", nbytes/(1024.f*1024.f) );
	h_data = (float*)malloc( nbytes );
	if( 0 == h_data )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
	printf("allocated %.2f MB on CPU\n", nbytes/(1024.f*1024.f) );
	for(int i=0; i<dimx*dimy; i++)
		h_data[i] = 10.f + rand() % 256;
	cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );

	float elapsed_time_ms=0.0f;

	elapsed_time_ms = timing_experiment( kernel_A, d_data, dimx,dimy, niterations, nreps, 32, 32);
	printf("A:  %8.2f ms\n", elapsed_time_ms );

	printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ) );

	if( d_data )
		cudaFree( d_data );
	if( h_data )
		free( h_data );

	cudaDeviceReset();

	return 0;
}


