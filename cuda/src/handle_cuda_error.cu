saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
cudaError_t errSync  = cudaGetLastError();
cudaError_t errAsync = cudaDeviceSynchronize();
if (errSync != cudaSuccess) 
  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync);
if (errAsync != cudaSuccess)
  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync);