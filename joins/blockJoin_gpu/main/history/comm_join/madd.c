#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define SHOW_TIME

int cuda_test_madd(unsigned int n, char *path)
{
  int i, j, idx;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function;
  CUmodule module;
  CUdeviceptr a_dev, b_dev, c_dev;
  unsigned int *a = (unsigned int *) malloc (n*n * sizeof(unsigned int));
  unsigned int *b = (unsigned int *) malloc (n*n * sizeof(unsigned int));
  unsigned int *c = (unsigned int *) malloc (n*n * sizeof(unsigned int));
  int block_x, block_y, grid_x, grid_y;
  char fname[256];

  struct timeval tv_HtoD_s, tv_HtoD_f, tv_cal_s, tv_cal_f, tv_DtoH_s, tv_DtoH_f;
  double time_HtoD, time_cal, time_DtoH;


  
  /* initialize A[] & B[] */
  for (i = 0; i < n; i++) {
    for(j = 0; j < n; j++) {
      idx = i * n + j;
      a[idx] = i;
      b[idx] = i + 1;
    }
  }

  /* block_x * block_y should not exceed 512. */
  block_x = n < 16 ? n : 16;
  block_y = n < 16 ? n : 16;
  grid_x = n / block_x;
  if (n % block_x != 0)
    grid_x++;
  grid_y = n / block_y;
  if (n % block_y != 0)
    grid_y++;

  /*この辺で初期化*/

  res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    printf("cuInit failed: res = %lu\n", (unsigned long)res);
    return -1;
  }

  res = cuDeviceGet(&dev, 0);
  if (res != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
    return -1;
  }

  res = cuCtxCreate(&ctx, 0, dev);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
    return -1;
  }


  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *
   */
  sprintf(fname, "%s/madd_gpu.cubin", path);
  res = cuModuleLoad(&module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    return -1;
  }
  res = cuModuleGetFunction(&function, module, "add");
  //  res = cuModuleGetFunction(&function, module, "_Z3addPjS_S_j");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction() failed\n");
    return -1;
  }
  res = cuFuncSetSharedSize(function, 0x40); /* just random */
  if (res != CUDA_SUCCESS) {
    printf("cuFuncSetSharedSize() failed\n");
    return -1;
  }


  /*ここからcuLaunchKernel代用*/
  res = cuFuncSetBlockShape(function, block_x, block_y, 1);
  if (res != CUDA_SUCCESS) {
    printf("cuFuncSetBlockShape() failed\n");
    return -1;
  }

  /*ここまでcuLaunchKernel代用*/
  
  /*
   *a,b,cのメモリを割り当てる。
   *
   */
  /* a[] */
  res = cuMemAlloc(&a_dev, n*n * sizeof(unsigned int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (a) failed\n");
    return -1;
  }
  /* b[] */
  res = cuMemAlloc(&b_dev, n*n * sizeof(unsigned int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (b) failed\n");
    return -1;
  }
  /* c[] */
  res = cuMemAlloc(&c_dev, n*n * sizeof(unsigned int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (c) failed\n");
    return -1;
  }
  
  /* upload a[] and b[] */
  gettimeofday(&tv_HtoD_s, NULL);
  res = cuMemcpyHtoD(a_dev, a, n*n * sizeof(unsigned int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (a) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuMemcpyHtoD(b_dev, b, n*n * sizeof(unsigned int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (b) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  gettimeofday(&tv_HtoD_f, NULL);
  


  /*ここからcuLaunchKernel代用*/
  /* set kernel parameters */
  res = cuParamSeti(function, 0, a_dev);	
  if (res != CUDA_SUCCESS) {
    printf("cuParamSeti (a) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuParamSeti(function, 4, a_dev >> 32);
  if (res != CUDA_SUCCESS) {
    printf("cuParamSeti (a) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuParamSeti(function, 8, b_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuParamSeti (b) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuParamSeti(function, 12, b_dev >> 32);
  if (res != CUDA_SUCCESS) {
    printf("cuParamSeti (b) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuParamSeti(function, 16, c_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuParamSeti (c) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuParamSeti(function, 20, c_dev >> 32);
  if (res != CUDA_SUCCESS) {
    printf("cuParamSeti (c) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuParamSeti(function, 24, n);
  if (res != CUDA_SUCCESS) {
    printf("cuParamSeti (c) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuParamSetSize(function, 28);
  if (res != CUDA_SUCCESS) {
    printf("cuParamSetSize failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  
  gettimeofday(&tv_cal_s, NULL);
  
  /* launch the kernel */
  res = cuLaunchGrid(function, grid_x, grid_y);
  if (res != CUDA_SUCCESS) {
    printf("cuLaunchGrid failed: res = %lu\n", (unsigned long)res);
    return -1;
  }

  /*ここまでcuLaunchKernel代用*/


  cuCtxSynchronize();
  
  gettimeofday(&tv_cal_f, NULL);
  
  /* download c[] */
  gettimeofday(&tv_DtoH_s, NULL);
  res = cuMemcpyDtoH(c, c_dev, n*n * sizeof(unsigned int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (c) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }

  gettimeofday(&tv_DtoH_f, NULL);

  res = cuMemFree(a_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (a) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuMemFree(b_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (b) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  res = cuMemFree(c_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (c) failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  
  res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  
  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
    return -1;
  }
  
  /* check the results */
  i = j = idx = 0;
  while (i < n) {
    j=0;
    while (j < n) {
      idx = i * n + j;
      if (c[idx] != a[idx] + b[idx]) {
	printf("c[%d] = %d\n", idx, c[idx]);
	printf("a[%d]+b[%d] = %d\n", idx, idx, a[idx]+b[idx]);
	return -1;
      }
      j++;
    }
    i++;
  }
  
#ifdef SHOW_TIME
  printf("\n");
  
  time_HtoD = (tv_HtoD_f.tv_sec - tv_HtoD_s.tv_sec)*1000*1000 + (tv_HtoD_f.tv_usec - tv_HtoD_s.tv_usec);
  printf("Memory copy Host to Devise : %6f(micro sec)\n", time_HtoD);
  
  time_cal = (tv_cal_f.tv_sec - tv_cal_s.tv_sec)*1000*1000 + (tv_cal_f.tv_usec - tv_cal_s.tv_usec);
  printf("Calculation with Devise    : %6f(micro sec)\n", time_cal);
  
  time_DtoH = (tv_DtoH_f.tv_sec - tv_DtoH_s.tv_sec)*1000*1000 + (tv_DtoH_f.tv_usec - tv_DtoH_s.tv_usec);
  printf("Memory copy Devise to Host : %6f(micro sec)\n", time_DtoH);
  
  printf("\n");

#endif




  free(a);
  free(b);
  free(c);
	
  return 0;
}


