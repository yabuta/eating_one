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

TUPLE *rt;
TUPLE *lt;
RESULT *jt;


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

  //メモリ割り当てを行う
  //タプルに初期値を代入
  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て****************************
  res = cuMemHostAlloc((void**)&rt,right * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to RIGHT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  //rt = (TUPLE *)malloc(right*sizeof(TUPLE));

  for (int i = 0; i < right; i++) {
    rt[i].key = getTupleId();
    rt[i].val = 1; // selectivity = 1.0
  }
  /*
  srand((unsigned)time(NULL));
  uint *used;
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  uint diff;
  if(MATCH_RATE != 0){
    diff = 1/MATCH_RATE;
  }else{
    diff = 1;
  }

  for (int i = 0; i < right; i++) {
    rt[i].key = getTupleId();
    uint temp = rand()%SELECTIVITY;
    while(used[temp] == 1) temp = rand()%SELECTIVITY;
    used[temp] = 1;
    rt[i].val = temp; // selectivity = 1.0
  }
*/

  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て*******************************
  res = cuMemHostAlloc((void**)&lt,left * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  //lt = (TUPLE *)malloc(left*sizeof(TUPLE));

  for (uint i = 0; i < left; i++) {
    lt[i].key = getTupleId();
    lt[i].val = 1; // selectivity = 1.0
  }

  /*
  uint l_diff;
  if(MATCH_RATE != 0){
    l_diff = left/(MATCH_RATE*right);
  }else{
    l_diff = 1;
  }
  for (uint i = 0; i < left; i++) {
    lt[i].key = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*right){
      lt[i].val = rt[counter*diff].val;
      counter++;
    }else{
      uint temp = rand()%SELECTIVITY;
      while(used[temp] == 1) temp = rand()%SELECTIVITY;
      lt[i].val = temp; // selectivity = 1.0
    }
  }
  free(used);
  */

  /*
  res = cuMemHostAlloc((void**)&jt,JT_SIZE * sizeof(RESULT),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */
  jt = (RESULT *)malloc(JT_SIZE*sizeof(RESULT));


}


/*      memory free           */
void freeTuple(){

  cuMemFreeHost(rt);
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

  /*
  sprintf(fname, "%s/join_gpu.cubin", path);
  res = cuModuleLoad(&module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&function, module, "join");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction() failed\n");
    exit(1);
  }
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

  gettimeofday(&time_index_s, NULL);
  //createIndex();
  gettimeofday(&time_index_f, NULL);

  gettimeofday(&begin, NULL);
  /****************************************************************/

  block_x = left < BLOCK_SIZE_X ? left : BLOCK_SIZE_X;
  grid_x = left / block_x;
  if (left % block_x != 0)
    grid_x++;

  /*
  block_y = right < BLOCK_SIZE_Y ? right : BLOCK_SIZE_Y;
  grid_y = right / block_y;
  if (right % block_y != 0)
    grid_y++;
  */
  block_y = 1;
  grid_y = 1;

  /********************************************************************
   *lt,rt,countのメモリを割り当てる。
   *
   */
  gettimeofday(&time_alloc_s, NULL);

  /* lt */
  res = cuMemAlloc(&lt_dev, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lefttuple) failed\n");
    exit(1);
  }
  /* rt */
  res = cuMemAlloc(&rt_dev, right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (righttuple) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&jt_dev, left * sizeof(RESULT));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (righttuple) failed\n");
    exit(1);
  }


  gettimeofday(&time_alloc_f, NULL);



  /**********************************************************************************/


  
  /********************** upload lt , rt , bucket ,buck_array ,idxcount***********************/

  gettimeofday(&time_send_s, NULL);
  gettimeofday(&time_tsend_s, NULL);

  res = cuMemcpyHtoD(lt_dev, lt, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (lt) failed: res = %lu\n", res);//conv(res));
    exit(1);
  }
  /*
  res = cuMemcpyHtoD(rt_dev, rt, right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */
  gettimeofday(&time_tsend_f, NULL);
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

  res = cuMemcpyDtoH(jt,jt_dev,left * sizeof(RESULT));
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
  /*
  res = cuMemFree(rt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  */

  res = cuMemFree(jt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (jointuple) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  } 
  
  printf("************index create time**************\n");
  printDiff(time_index_s,time_index_f);
  printf("\n");


  printf("\n************execution time****************\n\n");
  printf("all time:\n");
  printDiff(begin, end);
  printf("\n");
  printf("gpu memory alloc time:\n");
  printDiff(time_alloc_s,time_alloc_f);
  printf("\n");
  printf("data send time:\n");
  printDiff(time_send_s,time_send_f);
  /*
  printf("table data send time:\n");
  printDiff(time_tsend_s,time_tsend_f);
  */
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


  if(argc>3){
    printf("引数が多い\n");
    return 0;
  }else if(argc<3){
    printf("引数が足りない\n");
    return 0;
  }else{
    left=atoi(argv[1]);
    right=atoi(argv[2]);

    printf("left=%d:right=%d\n",left,right);
  }

  join();

  return 0;
}
