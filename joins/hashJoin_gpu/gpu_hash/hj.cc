#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/time.h>
#include <cuda.h>
#include <thrust/scan.h>
#include <math.h>
#include <ctype.h>
#include "debug.h"
#include "tuple.h"


TUPLE *rt;
TUPLE *lt;
RESULT *jt;

TUPLE *hrt;
TUPLE *hlt;

int rlocation[NB_BKTENT];
int rcount[NB_BKTENT];
int llocation[NB_BKTENT];
int lcount[NB_BKTENT];

int right,left;

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

  for (int i = 0; i < right; i++) {
    if(&(rt[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(rt[i]),0,sizeof(TUPLE));
    rt[i].key = getTupleId();

    for(int j = 0;j<NUM_VAL;j++){
      rt[i].val = rand()%SELECTIVITY; // selectivity = 1.0
      //rt[i].val = 1; // selectivity = 1.0
    }

  }

  /****************************************************************************/



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
    
    //０に初期化
    memset(&(lt[i]),0,sizeof(TUPLE));    
    lt[i].key = getTupleId();

    for(int j = 0; j < NUM_VAL;j++){
      lt[i].val = rand()%SELECTIVITY; // selectivity = 1.0
      //lt[i].val = 1; // selectivity = 1.0
    }

  }
  
  /*********************************************************************************/


  //JOIN_TUPLEへのGPUでも参照できるメモリの割り当て********************************
  res = cuMemHostAlloc((void**)&jt, JT_SIZE * sizeof(RESULT),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to JOIN_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  for(int i=0;i < JT_SIZE;i++){
    if(&(jt[i])==NULL){
      printf("left TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(jt[i]),0,sizeof(RESULT));
  }

  /**********************************************************************************/


}


/*      memory free           */
void freeTuple(){

  cuMemFreeHost(rt);
  cuMemFreeHost(lt);
  cuMemFreeHost(jt);

}


void
join()
{

  TUPLE *plt,*prt;
  RESULT result;
  int resultVal = 0;

  int p_num,t_num;
  int *l_p,*radix_num,*p_loc;
  int *count,*lL,*rL,*p_sum;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function,cp_function,p_function;
  CUmodule module,c_module,p_module,cp_module;
  CUdeviceptr lt_dev, rt_dev, jt_dev,count_dev,lL_dev,rL_dev;
  CUdeviceptr plt_dev,prt_dev,l_p_dev,r_p_dev,radix_dev;
  unsigned int block_x, block_y,p_block_x, grid_x, grid_y,p_grid_x;
  char fname[256];
  const char *path=".";
  struct timeval begin, end;
  struct timeval tv_cal_s, tv_cal_f,time_join_s,time_join_f,time_jkernel_s,time_jkernel_f,time_jdown_s,time_jdown_f;
  struct timeval time_count_s,time_count_f,time_alloc_s,time_alloc_f;
  struct timeval time_hash_s,time_hash_f;
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
    printf("cuModuleLoad(join) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&function, module, "join");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(join) failed\n");
    exit(1);
  }

  sprintf(fname, "%s/count.cubin", path);
  res = cuModuleLoad(&c_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(count) failed\n");
    exit(1);
  }
  
  res = cuModuleGetFunction(&c_function, c_module, "count");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(count) failed\n");
    exit(1);
  }

  sprintf(fname, "%s/count_partitioning.cubin", path);
  res = cuModuleLoad(&cp_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(count_partitioning) failed\n");
    exit(1);
  }
  
  res = cuModuleGetFunction(&cp_function, cp_module, "count_partitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(count_partitioning) failed\n");
    exit(1);
  }

  sprintf(fname, "%s/partitioning.cubin", path);
  res = cuModuleLoad(&p_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(partitioning) failed\n");
    exit(1);
  }
  
  res = cuModuleGetFunction(&p_function, p_module, "partitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(partitioning) failed\n");
    exit(1);
  }


  createTuple();

  /*
  for(int i = 0; i < left; i++){
    printf("%d = %d\n",i,lt[i].val);
  }
  */


  /*******************
   send data:
   int *lL,*rL
   TUPLE *rt
   TUPLE *lt
   RESULT *jt

  ******************/


  gettimeofday(&begin, NULL);

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


  /**********************************************************************************/

  gettimeofday(&time_alloc_f, NULL);


  /*preparation of partitioning*/


  gettimeofday(&time_hash_s, NULL);

  p_num = 0;
  double temp = left/B_ROW_NUM;

  if(temp < 2){
    p_num = 1;
  }else if(floor(log2(temp))==ceil(log2(temp))){
    p_num = (int)temp;
  }else{
    p_num = pow(2,(int)log2(temp) + 1);
  }


  t_num = left/PER_TH;
  if(left%PER_TH != 0){
    t_num++;
  }

  /*L */

  lL = (int *)calloc(p_num * t_num,sizeof(int));

  plt = (TUPLE *)calloc(left,sizeof(TUPLE));
  prt = (TUPLE *)calloc(right,sizeof(TUPLE));

  /*lL, plt and prt alloc in GPU */
  res = cuMemAlloc(&lL_dev, p_num * t_num * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lL) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&plt_dev, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (plt) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&prt_dev, right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (prt) failed\n");
    exit(1);
  }

  printf("t_num=%d\tp_num=%d\n",t_num,p_num);
  
  /********************** upload lt , rt , count ,plt, prt, rL, lL***********************/



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

  res = cuMemcpyHtoD(lL_dev, lL, p_num * t_num * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (lL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemcpyHtoD(plt_dev, plt, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (plt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemcpyHtoD(prt_dev, prt, right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (prt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  /***************************************************************************/



  /****************************************************************
    left table partitioning for hash


  ***************************************************************/

  p_block_x = t_num < PART_C_NUM ? t_num : PART_C_NUM;
  p_grid_x = t_num / p_block_x;
  if (t_num % p_block_x != 0)
    p_grid_x++;


  void *count_lpartition_args[]={
    
    (void *)&lt_dev,
    (void *)&lL_dev,
    (void *)&p_num,
    (void *)&t_num,
    (void *)&left
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       cp_function,    // CUfunction f
                       p_grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       p_block_x,       // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       count_lpartition_args,   // keunelParams
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


  res = cuMemcpyDtoH(lL, lL_dev, t_num * p_num * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (lL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  /*
  for(int i = 0; i<t_num*p_num ;i++){
     printf("%d = %d\n",i,lL[i]);

  }
  */


  /**************************** prefix sum *************************************/

  thrust::inclusive_scan(lL,lL + t_num*p_num,lL);

  /********************************************************************/

  for(int i = t_num*p_num-1; i>0 ;i--){
    lL[i] = lL[i-1];
  }
  lL[0] = 0;

  p_sum = (int *)calloc(p_num,sizeof(int));

  p_sum[0] = lL[t_num];
  for(int i = 1; i<p_num-1 ;i++){
    p_sum[i] = lL[t_num * (i+1)] - lL[t_num * i];
  }
  p_sum[p_num-1] = left - lL[t_num*(p_num-1)];

  

  /*
  for(int i = 0; i<t_num*p_num ;i++){
     printf("presum lL = %d\n",lL[i]);
  }

  for(int i=0; i<p_num ; i++){
    printf("p sum = %d\n",p_sum[i]);
  }
  */



  /***********************************************
    resizing partition 

  ************************************************/
  int l_p_num = 0;
  for(int i=0 ; i<p_num ;i++ ){

    if(p_sum[i]%B_ROW_NUM == 0 && p_sum[i]!=0){
      l_p_num += p_sum[i]/B_ROW_NUM;
    }else{
      l_p_num += p_sum[i]/B_ROW_NUM + 1;
    }            
  }


  l_p = (int *)calloc(l_p_num+1,sizeof(int));
  radix_num = (int *)calloc(l_p_num+1,sizeof(int));
  p_loc = (int *)calloc(p_num,sizeof(int));

  l_p_num = 0;
  int temp2 = 0;


  thrust::inclusive_scan(p_sum,p_sum + p_num,p_loc);

  for(int i = p_num-1; i>0 ;i--){
    p_loc[i] = p_loc[i-1];
  }
  p_loc[0] = 0;

  for(int i=0; i<p_num; i++){
    //printf("p_sum=%d\t%d\n",p_sum[i],B_ROW_NUM);
    if(p_sum[i]/B_ROW_NUM < 1 || p_sum[i]==B_ROW_NUM){
      l_p[l_p_num] = p_loc[i];
      radix_num[l_p_num] = i;
      l_p_num++;
    }else{
      if(p_sum[i]%B_ROW_NUM == 0){
        temp2 = p_sum[i]/B_ROW_NUM;
      }else{
        temp2 = p_sum[i]/B_ROW_NUM + 1;
      }
      l_p[l_p_num] = p_loc[i];
      radix_num[l_p_num] = i;
      l_p_num++;
      for(int j=1 ; j<temp2 ; j++){
        l_p[l_p_num] = p_loc[i]+j*B_ROW_NUM;
        radix_num[l_p_num] = i;
        l_p_num++;
      }
    }
  }

  l_p[l_p_num] = left;
  radix_num[l_p_num] = p_num;

  /*
  for(int i=0 ; i<l_p_num+1 ;i++){
    printf("%d\t%d\t%d\n",i,l_p[i],radix_num[i]);
  }
  */






  /***************************
   end of resizing partition

  ***********************************/

  res = cuMemcpyHtoD(lL_dev, lL, p_num * t_num * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (lL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  void *lpartition_args[]={
    
    (void *)&lt_dev,
    (void *)&plt_dev,
    (void *)&lL_dev,
    (void *)&p_num,
    (void *)&t_num,
    (void *)&left
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       p_function,    // CUfunction f
                       p_grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       p_block_x,       // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       lpartition_args,   // keunelParams
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
  

  res = cuMemcpyDtoH(plt, plt_dev, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (plt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }



  /**************************************************************/


  /*
  for(int i=0 ;i<left ;i++){
    printf("%d = %d\n",i,plt[i].val);
  }
  */



  //exit(1);


  /****************************************************************
    right table partitioning for hash


  ***************************************************************/


  
  t_num = right/PER_TH;
  if(right%PER_TH != 0){
    t_num++;
  }

  printf("t_num=%d\tp_num=%d\n",t_num,p_num);
  
  res = cuMemAlloc(&rL_dev, p_num * t_num * sizeof(int));
  if (res != CUDA_SUCCESS){
    printf("cuMemAlloc (rL) failed\n");
    exit(1);
  }

  rL = (int *)calloc(p_num * t_num,sizeof(int));

  res = cuMemcpyHtoD(rL_dev, rL, p_num * t_num * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  p_block_x = t_num < PART_C_NUM ? t_num : PART_C_NUM;
  p_grid_x = t_num / p_block_x;
  if (t_num % p_block_x != 0)
    p_grid_x++;


  void *count_rpartition_args[]={
    
    (void *)&rt_dev,
    (void *)&rL_dev,
    (void *)&p_num,
    (void *)&t_num,
    (void *)&right
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       cp_function,    // CUfunction f
                       p_grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       p_block_x,       // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       count_rpartition_args,   // keunelParams
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


  res = cuMemcpyDtoH(rL, rL_dev, t_num * p_num * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (rL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  /**************************** prefix sum *************************************/

  thrust::inclusive_scan(rL,rL + t_num*p_num,rL);

  /********************************************************************/

  
  for(int i = t_num*p_num-1; i>0 ;i--){
    rL[i] = rL[i-1];
  }
  rL[0] = 0;


  int *r_p;
  r_p = (int *)calloc(p_num+1,sizeof(int));

  for(int i = 0; i<p_num ;i++){
    r_p[i] = rL[t_num * i];
  }
  r_p[p_num] = right;


  /*
  for(int i=0 ; i<p_num+1 ;i++){
    printf("%d\n",r_p[i]);
  }
  */


  res = cuMemcpyHtoD(rL_dev, rL, p_num * t_num * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  void *rpartition_args[]={
    
    (void *)&rt_dev,
    (void *)&prt_dev,
    (void *)&rL_dev,
    (void *)&p_num,
    (void *)&t_num,
    (void *)&right
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       p_function,    // CUfunction f
                       p_grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       p_block_x,       // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       rpartition_args,   // keunelParams
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


  res = cuMemcpyDtoH(prt, prt_dev, right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (prt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  /*
  for(int i = 0; i < right; i++){
    printf("right:%d = %d\n",i,prt[i].val);
  }
  */




  gettimeofday(&time_hash_f, NULL);

  /**************************************************************/


  /**
     plt and prt is created from lt and rt
     l_p and radix_num is left table location of each partition for read 
     r_p is right table location of each partition

     we use these value for nest loop join of each same partition of both table

   **/



  /******************************************************************
    count the number of match tuple
    
  *******************************************************************/

  gettimeofday(&time_count_s, NULL);


  block_x = right < BLOCK_SIZE_X ? right : BLOCK_SIZE_X;
  block_y = 1;//right < BLOCK_SIZE_Y ? right : BLOCK_SIZE_Y;

  grid_x = l_p_num;


  /*GPU memory alloc and send data of count ,l_p ,radix and r_p*/
  res = cuMemAlloc(&count_dev, grid_x * block_x * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&l_p_dev, (l_p_num+1) * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (l_p) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&radix_dev, (l_p_num+1) * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (radix) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&r_p_dev, (p_num+1) * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (r_p) failed\n");
    exit(1);
  }


  count = (int *)calloc(grid_x*block_x*block_y,sizeof(int));

  res = cuMemcpyHtoD(count_dev, count, grid_x * block_x * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemcpyHtoD(l_p_dev, l_p, (l_p_num+1) * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemcpyHtoD(radix_dev, radix_num, (l_p_num+1) * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemcpyHtoD(r_p_dev, r_p, (p_num+1) * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  /*
    lt_dev       left table
    rt_dev       right table
    count_dev    the number of matched tuples per thread
    r_p_dev      location of right table partition begin
    radix_dev    radix of left table partition
    l_p_dev    location of left table partition begin

   */

  void *count_args[]={
    
    (void *)&plt_dev,
    (void *)&prt_dev,
    (void *)&count_dev,
    (void *)&r_p_dev,
    (void *)&radix_dev,
    (void *)&l_p_dev,
    (void *)&right,
    (void *)&left
      
  };

  int sharedMemBytes = B_ROW_NUM*sizeof(TUPLE);


  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       c_function,    // CUfunction f
                       grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       block_x,     // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       sharedMemBytes,             // sharedMemBytes
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


  res = cuMemcpyDtoH(count, count_dev, grid_x * block_x * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }



  /**************************** prefix sum *************************************/

  thrust::inclusive_scan(count,count + grid_x*block_x*block_y,count);

  /********************************************************************/

  int jt_size = count[grid_x*block_x*block_y-1];

  for(int i=grid_x*block_x*block_y-1 ; i>0 ;i--){
    count[i] = count[i-1];

  }
  count[0] = 0;



  printf("%d\t%d\n",jt_size,grid_x*block_x*block_y);
  /*
  for(int i=0 ; i<grid_x*block_x*block_y ; i++){
    printf("%d\t%d\n",i,count[i]);

  }
  */



  gettimeofday(&time_count_f, NULL);



  /***************************************************************************************/





  /************************************************************************
   p memory alloc and p upload
   supplementary:
   The reason that join table is "p" ,it's used sample program .
   As possible I will change appriciate value.
  ************************************************************************/

  if(jt_size <= 0){
    printf("no tuple is matched.\n");
    exit(1);
  }else{
    res = cuMemAlloc(&jt_dev, jt_size * sizeof(RESULT));
    if (res != CUDA_SUCCESS) {
      printf("cuMemAlloc (jt) failed\n");
      exit(1);
    }
  }


  gettimeofday(&time_join_s, NULL);


  //gettimeofday(&time_jup_s, NULL);

  res = cuMemcpyHtoD(jt_dev, jt, jt_size * sizeof(RESULT));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (join) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemcpyHtoD(count_dev, count, grid_x * block_x * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  //gettimeofday(&time_jup_f, NULL);

  gettimeofday(&time_jkernel_s, NULL);

  void *kernel_args[]={
    (void *)&plt_dev,
    (void *)&prt_dev,
    (void *)&jt_dev,
    (void *)&count_dev,
    (void *)&r_p_dev,
    (void *)&radix_dev,
    (void *)&l_p_dev,
    (void *)&left,
    (void *)&right,    
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
  res = cuLaunchKernel(
                       function,      // CUfunction f
                       grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       block_x,       // blockDimX
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


  /********************************************************************/

  gettimeofday(&time_join_f, NULL);


  gettimeofday(&end, NULL);


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

  res = cuMemFree(count_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  res = cuMemFree(jt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (jointuple) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  res = cuMemFree(plt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (plt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemFree(prt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (prt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  res = cuMemFree(lL_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  res = cuMemFree(rL_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  free(lL);
  free(rL);
  free(plt);
  free(prt);
  free(p_sum);
  free(l_p);
  free(radix_num);
  free(p_loc);



  printf("all time:\n");
  printDiff(begin, end);
  printf("memory allocate time:\n");
  printDiff(time_alloc_s,time_alloc_f);
  printf("hash time:\n");
  printDiff(time_hash_s,time_hash_f);
  printf("count time:\n");
  printDiff(time_count_s,time_count_f);
  printf("join time:\n");
  printDiff(time_join_s,time_join_f);
  printf("kernel launch time of join:\n");
  printDiff(time_jkernel_s,time_jkernel_f);
  printf("join download time:\n");
  printDiff(time_jdown_s,time_jdown_f);

  /*
  printf("count time:\n");
  printDiff(time_count_s,time_count_f);
  printf("upload time of jt:\n");
  printDiff(time_jup_s,time_jup_f);
  printf("download time of jt:\n");
  printDiff(time_jdown_s,time_jdown_f);
  */

  int temp3=0;

  for(int i=0 ; i<left ;i++ ){
    if(lt[i].val==0){
      temp3++;
    }
  }
  printf("lt0 = %d\n",temp3);
  temp3=0;

  for(int i=0 ; i<right ;i++ ){
    if(rt[i].val==0){
      temp3++;
    }
  }
  printf("rt0 = %d\n",temp3);
  temp3=0;

  for(int i=0 ; i<right ;i++ ){
    if(rt[i].val==1){
      temp3++;
    }
  }
  printf("rt1 = %d\n",temp3);
  temp3=0;

  for(int i=0 ; i<left ;i++ ){
    if(plt[i].val==0){
      temp3++;
    }
  }
  printf("plt0 = %d\n",temp3);
  temp3=0;

  for(int i=0 ; i<right ;i++ ){
    if(prt[i].val==0){
      temp3++;
    }
  }
  printf("prt0 = %d\n",temp3);
  temp3=0;

  for(int i=0 ; i<right ;i++ ){
    if(prt[i].val!=0&&prt[i].val!=1){
      printf("%d\t%d\n",i,prt[i].val);
      temp3++;
    }
  }
  printf("prt1 = %d\n",temp3);
  temp3=0;

  for(int i=0; i<right ;i++){
    for(int j=0 ;j<right ;j++){
      if(prt[i].key==rt[j].key&&prt[i].val != rt[j].val){
        printf("diff rt and prt = %d\t%d\t%d\t%d\t%d\n",i,prt[i].key,prt[i].val,rt[j].key,prt[j].val);
      }
    }
  }

  for(int i=0 ; i<left ; i++){
    for(int j=0; j<right ;j++){
      if(plt[i].val == prt[j].val){
        temp3++;
      }
    }
  }

  printf("p sequence result = %d\n",temp3);
  temp3=0;

  for(int i=0 ; i<left ; i++){
    for(int j=0; j<right ;j++){
      if(lt[i].val == rt[j].val){
        temp3++;
      }
    }
  }

  printf("normal sequence result = %d\n",temp3);



  /*
  printf("result of join tuple----------\n");
  for(int i=0; i<jt_size ; i++){
    if(i%10000==0){
      printf("left:id = %d  val = %d\tright:id=%d  val = %d\n",jt[i].lkey,jt[i].lval,jt[i].rkey,jt[i].rval);
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
