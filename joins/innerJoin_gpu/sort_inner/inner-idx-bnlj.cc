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
#include "debug.h"
#include "scan_common.h"
#include "tuple.h"

BUCKET *Bucket;
int Buck_array[NB_BKT_ENT];
int idxcount[NB_BKT_ENT];

IDX Hidx;

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

  srand((unsigned)time(NULL));
  uint *used;
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  for (int i = 0; i < right; i++) {
    if(&(rt[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }
    rt[i].key = getTupleId();
    if(i < right*MATCH_RATE){
      uint temp = rand()%SELECTIVITY;
      while(used[temp] == 1) temp = rand()%SELECTIVITY;
      used[temp] = 1;
      rt[i].val = temp; // selectivity = 1.0
    }else{
      rt[i].val = SELECTIVITY + rand()%SELECTIVITY;
    }
  }

  free(used);



  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て*******************************
  res = cuMemHostAlloc((void**)&lt,left * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  for (int i = 0; i < left; i++) {
    if(&(lt[i])==NULL){
      printf("left TUPLE allocate error.\n");
      exit(1);
    }    
    lt[i].key = getTupleId();
    if(i < right * MATCH_RATE){
      lt[i].val = rt[i].val; // selectivity = 1.0
    }else{
      lt[i].val = 2 * SELECTIVITY + rand()%SELECTIVITY;
    }
  }

}


/*      memory free           */
void freeTuple(){

  cuMemFreeHost(rt);
  cuMemFreeHost(lt);
  cuMemFreeHost(jt);
  free(Bucket);

}



/***************create index*****************************/
/*
  swap(TUPLE *a,TUPLE *b)
  partition(int p,int q,int *jp,int *ip)
  qsort(int p,int q)
  createIndex(void)
*/

void swap(BUCKET *a,BUCKET *b)
{
  BUCKET temp;
  temp.val = a->val;
  temp.adr = a->adr;
  a->val = b->val;
  a->adr = b->adr;
  b->val = temp.val;
  b->adr = temp.adr;

}

void partition(int p,int q,int *jp,int *ip)
{

  int i,j,s;
  while(i<=j){
    while(Bucket[i].val<s) ++i;
    while(Bucket[j].val>s) --j;
    if(i<=j){
      swap(&Bucket[i],&Bucket[j]);
      ++i;
      j--;
    }
  }
  *jp = j;
  *ip = i;

}

void qsort(int p,int q)
{
  int i,j;
  if(p<q){
    partition(p,q,&i,&j);
    qsort(p,j);
    qsort(i,p);
  }
}

void createIndex(void)
{

  if (!(Bucket = (BUCKET *)calloc(right, sizeof(BUCKET)))) ERR;
  for(uint i=0; i<right ; i++){
    Bucket[i].val = rt[i].val;
    Bucket[i].adr = i;
  }

  qsort(1,right);

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
  CUdeviceptr c_dev,temp_dev;
  unsigned int block_x, block_y, grid_x, grid_y;
  char fname[256];
  const char *path=".";
  struct timeval begin, end;
  struct timeval tv_cal_s, tv_cal_f,time_join_s,time_join_f,time_upload_s,time_upload_f,time_download_s,time_download_f;
  struct timeval time_count_s,time_count_f,time_tsend_s,time_tsend_f,time_isend_s,time_isend_f;
  struct timeval time_jdown_s,time_jdown_f,time_jkernel_s,time_jkernel_f;
  struct timeval time_scan_s,time_scan_f,time_alloc_s,time_alloc_f,time_cinit_s,time_cinit_f;
  double time_cal;




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
  createIndex();

  printf("ok\n");

  gettimeofday(&begin, NULL);
  /****************************************************************/

  block_y = left < BLOCK_SIZE_Y ? left : BLOCK_SIZE_Y;
  
  grid_y = left / block_y;
  if (right % block_y != 0)
    grid_y++;
  

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
  /* bucket */
  res = cuMemAlloc(&bucket_dev, right * sizeof(BUCKET));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (bucket) failed\n");
    exit(1);
  }

  /*count */
  res = cuMemAlloc(&c_dev, left * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }

  gettimeofday(&time_alloc_f, NULL);



  /**********************************************************************************/


  
  /********************** upload lt , rt , bucket ,buck_array ,idxcount***********************/


  gettimeofday(&time_tsend_s, NULL);

  res = cuMemcpyHtoD(lt_dev, lt, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (lt) failed: res = %lu\n", res);//conv(res));
    exit(1);
  }
  res = cuMemcpyHtoD(rt_dev, rt, right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  gettimeofday(&time_tsend_f, NULL);

  gettimeofday(&time_isend_s, NULL);

  res = cuMemcpyHtoD(bucket_dev, Bucket, right * sizeof(BUCKET));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (bucket) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  gettimeofday(&time_isend_f, NULL);


  /***************************************************************************/



  /*
    条件に合致するタプルの数をあらかじめ求めておく
    これによってダウンロードするタプルの数を減らせる
   */



  /******************************************************************
    count the number of match tuple
    
  *******************************************************************/

  gettimeofday(&time_count_s, NULL);


  void *count_args[]={
    
    (void *)&lt_dev,
    (void *)&c_dev,
    (void *)&bucket_dev,
    (void *)&right,
    (void *)&left
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       c_function,    // CUfunction f
                       1,        // gridDimX
                       grid_y,        // gridDimY
                       1,             // gridDimZ
                       1,       // blockDimX
                       block_y,       // blockDimY
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


  /***************************************************************************************/



  /**************************** prefix sum *************************************/
  gettimeofday(&time_scan_s, NULL);

  if(!(presum(&c_dev,(uint)left))){
    printf("count scan error\n");
    exit(1);
  }

  gettimeofday(&time_scan_f, NULL);

  /********************************************************************/

  
  gettimeofday(&time_count_f, NULL);



  /************************************************************************
   join

   jt memory alloc and jt upload
  ************************************************************************/

  gettimeofday(&time_join_s, NULL);

  if(!transport(c_dev,(uint)left,&jt_size)){
    printf("transport error.\n");
    exit(1);
  }
  if(jt_size <= 0){
    printf("no tuple is matched.\n");
    exit(1);
  }
  res = cuMemAlloc(&jt_dev, jt_size * sizeof(RESULT));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (join) failed\n");
    exit(1);
  }
  jt = (RESULT *)malloc(jt_size*sizeof(RESULT));
  gettimeofday(&time_jkernel_s, NULL);

  void *kernel_args[]={
    (void *)&rt_dev,
    (void *)&lt_dev,
    (void *)&jt_dev,
    (void *)&c_dev,
    (void *)&bucket_dev,
    (void *)&right,
    (void *)&left    
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
  res = cuLaunchKernel(
                       function,      // CUfunction f
                       1,        // gridDimX
                       grid_y,        // gridDimY
                       1,             // gridDimZ
                       1,       // blockDimX
                       block_y,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       kernel_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS) {
    printf("join cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  

  gettimeofday(&time_jkernel_f, NULL);

  gettimeofday(&time_jdown_s, NULL);

  res = cuMemcpyDtoH(jt, jt_dev, jt_size * sizeof(RESULT));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (p) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  gettimeofday(&time_jdown_f, NULL);


  gettimeofday(&time_join_f, NULL);


  gettimeofday(&end, NULL);



  /***************************************************************
  free GPU memory
  ***************************************************************/

  res = cuMemFree(lt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(rt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  res = cuMemFree(jt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (jointuple) failed: res = %lu\n", (unsigned long)res);
    exit(1);
    }

  res = cuMemFree(c_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(bucket_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (bucket) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  printf("\n************execution time****************\n\n");
  printf("all time:\n");
  printDiff(begin, end);
  printf("\n");
  printf("gpu memory alloc time:\n");
  printDiff(time_alloc_s,time_alloc_f);
  printf("\n");
  printf("table data send time:\n");
  printDiff(time_tsend_s,time_tsend_f);
  printf("\n");
  printf("index data send time:\n");
  printDiff(time_isend_s,time_isend_f);
  printf("\n");
  printf("count time:\n");
  printDiff(time_count_s,time_count_f);
  printf("count scan time:\n");
  printDiff(time_scan_s,time_scan_f);
  printf("\n");
  printf("join time:\n");
  printDiff(time_join_s,time_join_f);
  printf("kernel launch time of join:\n");
  printDiff(time_jkernel_s,time_jkernel_f);
  printf("download time of jt:\n");
  printDiff(time_jdown_s,time_jdown_f);


  printf("%d\n",jt_size);
  


  /*
  for(int i = 0; i < count[right - 1] ;i++){

    if(jt[i].lval == jt[i].rval && i % 100000==0){

      printf("lid=%d  left=%d\trid=%d  right=%d\n",jt[i].lkey,jt[i].lval,jt[i].rkey,jt[i].rval);
      //printf("left=%d\tright=%d\n",jt[i].lval,jt[i].rval);
    }
  }
  */


  //finish GPU   ****************************************************
  res = cuModuleUnload(module);
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
  }else if(argc<2){
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
