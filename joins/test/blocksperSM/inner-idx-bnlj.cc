#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
//#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
//#include "debug.h"
#include "tuple.h"

/*
int *lt;
int *jt;
*/
int *lt;
int *jt;

int left;

void
printDiff(struct timeval begin, struct timeval end)
{
  long diff;  
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

static int
getTupleId(void)
{
  static int id;
  return ++id;
}

void createTuple()
{

  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て
  CUresult res;
  struct timeval start,finish;
  //メモリ割り当てを行う
  //タプルに初期値を代入

  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て*******************************
  res = cuMemHostAlloc((void**)&lt,left * sizeof(int),CU_MEMHOSTALLOC_PORTABLE);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  //lt = (int *)malloc(left*sizeof(int));
  srand((unsigned)time(NULL));
  for (uint i = 0; i < left; i++) {
    //lt[i].key = getTupleId();
    lt[i] = i; // selectivity = 1.0
  }

  /*allocated cuMemHostAlloc or malloc array access time difference
  gettimeofday(&start, NULL);
  for (uint i = 0; i < left; i++) {
    int temp = lt[i] + 1; // selectivity = 1.0
  }
  gettimeofday(&finish, NULL);

  printf("CPU access time to pinned memory:\n");
  printDiff(start, finish);
  printf("\n");
  */

  res = cuMemHostAlloc((void**)&jt,left * sizeof(int),CU_MEMHOSTALLOC_PORTABLE);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  //  jt = (int *)malloc(left*sizeof(int));


}


/*      memory free           */
void freeTuple(){

  /*
  free(lt);
  free(jt);
  */
  cuMemFreeHost(lt);
  cuMemFreeHost(jt);

}

void join(){

  //uint *count;
  uint jt_size;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function;
  CUmodule module,c_module;
  CUdeviceptr lt_dev, rt_dev, jt_dev, bucket_dev, buckArray_dev ,idxcount_dev;
  CUdeviceptr ltn_dev, rtn_dev, jt_size_dev;
  CUdeviceptr c_dev;
  unsigned int block_x, grid_x,block_y,grid_y;
  char fname[256];
  const char *path=".";
  struct timeval begin, end;
  struct timeval time_join_s,time_join_f,time_send_s,time_send_f;
  struct timeval time_count_s,time_count_f,time_tsend_s,time_tsend_f,time_isend_s,time_isend_f;
  struct timeval time_jdown_s,time_jdown_f,time_jkernel_s,time_jkernel_f;
  struct timeval time_scan_s,time_scan_f,time_alloc_s,time_alloc_f,time_index_s,time_index_f;
  //double time_cal;




  /******************** GPU init here ************************************************/
  //GPU仕様のために

  res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    printf("cuInit failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuDeviceGet(&dev, 0);
  if (res != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuCtxCreate(&ctx, 0, dev);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  /*********************************************************************************/



  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *今回はjoin_gpu.cubinとcountJoinTuple.cubinの二つの関数を実行する
   */

  sprintf(fname, "%s/count.cubin", path);
  res = cuModuleLoad(&c_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    exit(1);
  }
  
  res = cuModuleGetFunction(&c_function, c_module, "count");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction() failed\n");
    exit(1);
  }

  /*tuple and index init******************************************/  

  createTuple();

  gettimeofday(&begin, NULL);
  /****************************************************************/

  block_x = left < BLOCK_SIZE_X ? left : BLOCK_SIZE_X;
  grid_x = left / block_x;
  if (left % block_x != 0)
    grid_x++;

  block_y = 1;
  grid_y = 1;

  /********************************************************************
   *lt,rt,countのメモリを割り当てる。
   *
   */
  gettimeofday(&time_alloc_s, NULL);

  /* lt */
  res = cuMemAlloc(&lt_dev, left * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lefttuple) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&jt_dev, left * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (righttuple) failed\n");
    exit(1);
  }


  gettimeofday(&time_alloc_f, NULL);



  /**********************************************************************************/


  
  /********************** upload lt , rt , bucket ,buck_array ,idxcount***********************/

  gettimeofday(&time_send_s, NULL);
  res = cuMemcpyHtoD(lt_dev, lt, left * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (lt) failed: res = %lu\n", res);//conv(res));
    exit(1);
  }
  gettimeofday(&time_send_f, NULL);



  /******************************************************************
    count the number of match tuple
    
  *******************************************************************/

  gettimeofday(&time_count_s, NULL);


  void *count_args[]={
    
    (void *)&lt_dev,
    (void *)&jt_dev,
    (void *)&left
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       c_function,    // CUfunction f
                       grid_x,        // gridDimX
                       grid_y,    // gridDimY
                       1,             // gridDimZ
                       block_x,       // blockDimX
                       block_y,  // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       count_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS) {
    printf("count cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }      

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  

  gettimeofday(&time_count_f, NULL);

  gettimeofday(&time_jdown_s, NULL);
  res = cuMemcpyDtoH(jt,jt_dev,left * sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (jt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  gettimeofday(&time_jdown_f, NULL);
  gettimeofday(&end, NULL);

  /***************************************************************
  free GPU memory
  ***************************************************************/

  res = cuMemFree(lt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemFree(jt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (jointuple) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  } 
  

  printf("\n************execution time****************\n\n");
  printf("all time:\n");
  printDiff(begin, end);
  printf("\n");
  printf("gpu memory alloc time:\n");
  printDiff(time_alloc_s,time_alloc_f);
  printf("\n");
  printf("data send time:\n");
  printDiff(time_send_s,time_send_f);
  printf("\n");
  printf("count time:\n");
  printDiff(time_count_s,time_count_f);
  printf("download time of jt:\n");
  printDiff(time_jdown_s,time_jdown_f);
  
  //finish GPU   ****************************************************
  res = cuModuleUnload(c_module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload module failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  
  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  /****************************************************************************/

  freeTuple();


}


int 
main(int argc,char *argv[])
{

  left = 1024*1024*256;
  printf("number = %d\n",left);

  join();

  return 0;
}
