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
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
//#include <thrust/scan.h>
#include <math.h>
#include <ctype.h>
#include "debug.h"
#include "scan_common.h"
#include "tuple.h"


TUPLE *rt;
TUPLE *lt;
RESULT *jt;

TUPLE *hrt;
TUPLE *hlt;

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
  uint *used;
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));

  for (uint i = 0; i < right; i++) {
    if(&(rt[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(rt[i]),0,sizeof(TUPLE));
    rt[i].key = getTupleId();

    if(i < right*MATCH_RATE/100){
      uint temp = rand()%SELECTIVITY;
      while(used[temp] == 1) temp = rand()%SELECTIVITY;
      used[temp] = 1;
      rt[i].val = temp; // selectivity in tuple.h
    }else{
      rt[i].val = SELECTIVITY + rand()%SELECTIVITY;
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

    //lt[i].val = rand()%SELECTIVITY; // selectivity in tuple.h

    if(i < right*MATCH_RATE/100){
      lt[i].val = rt[i].val; // selectivity in tuple.h
    }else{
      lt[i].val = 2 * SELECTIVITY + rand()%SELECTIVITY;//rand()%SELECTIVITY; // selectivity in tuple.h
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

  //TUPLE *plt,*prt;
  RESULT result;
  int resultVal = 0;

  uint p_num,t_num;
  uint r_p_max;
  uint *l_p,*radix_num,*p_loc;
  uint *count,*p_sum;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function,cp_function,p_function;
  CUmodule module,c_module,p_module,cp_module;
  CUdeviceptr lt_dev, rt_dev, jt_dev,count_dev,lL_dev,rL_dev;
  CUdeviceptr plt_dev,prt_dev,l_p_dev,r_p_dev,radix_dev,p_sum_dev;
  unsigned int block_x, block_y, grid_x, grid_y,p_grid_x,p_block_x;
  char fname[256];
  const char *path=".";
  struct timeval begin, end;
  struct timeval time_join_s,time_join_f,time_jkernel_s,time_jkernel_f;
  struct timeval time_jdown_s,time_jdown_f,time_upload_s,time_upload_f;
  struct timeval time_hash_s,time_hash_f,time_hkernel_s,time_hkernel_f,time_lhash_s,time_lhash_f,time_rhash_s,time_rhash_f,time_lscan_s,time_lscan_f,time_rscan_s,time_rscan_f,time_cscan_s,time_cscan_f;
  struct timeval time_lhck_s,time_lhck_f,time_rhck_s,time_rhck_f,time_rhk_s,time_rhk_f;
  struct timeval time_count_s,time_count_f,time_ckernel_s,time_ckernel_f,time_alloc_s,time_alloc_f;
  struct timeval temp_s,temp_f;
  double time_cal;
  long temper=0,tempest=0;
  long uptime = 0;




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
  res = cuModuleGetFunction(&c_function, module, "count");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(count) failed\n");
    exit(1);
  }
  sprintf(fname, "%s/partitioning.cubin", path);
  res = cuModuleLoad(&p_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(partitioning) failed\n");
    exit(1);
  }

  res = cuModuleGetFunction(&cp_function, p_module, "count_partitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(count_partitioning) failed\n");
    exit(1);
  }
  
  res = cuModuleGetFunction(&p_function, p_module, "partitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(partitioning) failed\n");
    exit(1);
  }


  createTuple();

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

  //lL = (uint *)calloc(p_num * t_num,sizeof(uint));

  printf("%d\n",p_num * t_num);

  /*lL, plt and prt alloc in GPU */
  res = cuMemAlloc(&lL_dev, p_num * t_num * sizeof(uint));
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


  gettimeofday(&time_upload_s, NULL);  

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

  gettimeofday(&time_upload_f, NULL);  

  /*
  checkCudaErrors(cudaMemset((void *)prt_dev, 0 , left*sizeof(TUPLE)));

  checkCudaErrors(cudaMemset((void *)plt_dev, 0 , right*sizeof(TUPLE)));
  */

  /***************************************************************************/



  /****************************************************************
    left table partitioning for hash


  ***************************************************************/

  gettimeofday(&time_lhash_s, NULL);

  p_block_x = t_num < PART_C_NUM ? t_num : PART_C_NUM;
  p_grid_x = t_num / p_block_x;
  if (t_num % p_block_x != 0)
    p_grid_x++;


  checkCudaErrors(cudaMemset((void *)lL_dev, 0 , p_num*t_num*sizeof(uint)));

  gettimeofday(&time_lhck_s, NULL);

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

  gettimeofday(&time_lhck_f, NULL);

  /**************************** prefix sum *************************************/

  gettimeofday(&time_lscan_s, NULL);

  if(!(presum(&lL_dev,(uint)t_num*p_num))){
    printf("lL presum error\n");
    exit(1);
  }

  gettimeofday(&time_lscan_f, NULL);

  /********************************************************************/

  p_sum = (uint *)calloc(p_num,sizeof(uint));

  if(!(p_sum_dev = diff_part(lL_dev,t_num,p_num,left))){
    printf("p_sum prefix sum error.\n");
    exit(1);
  }

  res = cuMemcpyDtoH(p_sum,p_sum_dev,p_num * sizeof(uint)); 
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (p_sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);    
  }


  /***********************************************
    resizing partition 

  ************************************************/
  uint l_p_num = 0;
  for(uint i=0 ; i<p_num ;i++ ){

    if(p_sum[i]%B_ROW_NUM == 0 && p_sum[i]!=0){
      l_p_num += p_sum[i]/B_ROW_NUM;
    }else{
      l_p_num += p_sum[i]/B_ROW_NUM + 1;
    }            
  }


  l_p = (uint *)calloc(l_p_num+1,sizeof(uint));
  radix_num = (uint *)calloc(l_p_num+1,sizeof(uint));
  p_loc = (uint *)calloc(p_num,sizeof(uint));

  l_p_num = 0;
  uint temp2 = 0;


  /****************presum*****************/
  p_loc[0] = 0;
  for(int i=1 ; i<p_num ; i++){
    p_loc[i] = p_loc[i-1] + p_sum[i-1];
  }  

  /***************************************/

  for(int i=0; i<p_num; i++){
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
      for(uint j=1 ; j<temp2 ; j++){
        l_p[l_p_num] = p_loc[i]+j*B_ROW_NUM;
        radix_num[l_p_num] = i;
        l_p_num++;
      }
    }
  }

  l_p[l_p_num] = left;
  radix_num[l_p_num] = p_num;

  /***************************
   end of resizing partition

  ***********************************/

  gettimeofday(&time_hkernel_s, NULL);

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
    printf("left partition cuCtxSynchronize() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  

  gettimeofday(&time_hkernel_f, NULL);  


  /**************************************************************/

  gettimeofday(&time_lhash_f, NULL);


  /****************************************************************
    right table partitioning for hash


  ***************************************************************/

  gettimeofday(&time_rhash_s, NULL);

  
  t_num = right/PER_TH;
  if(right%PER_TH != 0){
    t_num++;
  }

  printf("t_num=%d\tp_num=%d\n",t_num,p_num);

  res = cuMemAlloc(&rL_dev, p_num * t_num * sizeof(uint));
  if (res != CUDA_SUCCESS){
    printf("cuMemAlloc (rL) failed\n");
    exit(1);
  }

  checkCudaErrors(cudaMemset((void *)rL_dev, 0 , p_num*t_num*sizeof(uint)));

  p_block_x = t_num < PART_C_NUM ? t_num : PART_C_NUM;
  p_grid_x = t_num / p_block_x;
  if (t_num % p_block_x != 0)
    p_grid_x++;

  gettimeofday(&time_rhck_s, NULL);

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


  gettimeofday(&time_rhck_f, NULL);


  /**************************** prefix sum *************************************/

  gettimeofday(&time_rscan_s, NULL);

  if(!(presum(&rL_dev,(uint)t_num*p_num))){
    printf("presum error\n");
    exit(1);
  }

  gettimeofday(&time_rscan_f, NULL);

  /********************************************************************/

  uint *r_p =  (uint *)calloc(p_num+1,sizeof(uint));
  uint rdiff;

  if(!(r_p_dev = transport(rL_dev,t_num,p_num,right))){
    printf("transport error.\n");
    exit(1);
  }


  res = cuMemcpyDtoH(r_p,r_p_dev,(p_num+1) * sizeof(uint)); 
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (r_p) failed: res = %lu\n", (unsigned long)res);
    exit(1);    
  }

  for(uint i = 0; i<p_num+1 ;i++){
    if(i==0){
      r_p_max = r_p[i];
    }else{
      rdiff = r_p[i] - r_p[i-1];
      if(rdiff > r_p_max){
        r_p_max = rdiff;
      }
    }
  }

  gettimeofday(&time_rhk_s, NULL);

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

  gettimeofday(&time_rhk_f, NULL);

  gettimeofday(&time_rhash_f, NULL);

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


  block_x = r_p_max < BLOCK_SIZE_X ? r_p_max : BLOCK_SIZE_X;
  block_y = BLOCK_SIZE_Y;

  grid_x = l_p_num;
  grid_y = GRID_SIZE_Y;

  /*GPU memory alloc and send data of count ,l_p ,radix and r_p*/
  res = cuMemAlloc(&count_dev, grid_x * block_x * grid_y * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&l_p_dev, (l_p_num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (l_p) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&radix_dev, (l_p_num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (radix) failed\n");
    exit(1);
  }

  count = (uint *)calloc(3,sizeof(uint));


  checkCudaErrors(cudaMemset((void *)count_dev, 0 , grid_x*block_x*grid_y*sizeof(uint)));

  res = cuMemcpyHtoD(l_p_dev, l_p, (l_p_num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemcpyHtoD(radix_dev, radix_num, (l_p_num+1) * sizeof(uint));
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

  gettimeofday(&time_ckernel_s, NULL);

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
                       grid_y,        // gridDimY
                       1,             // gridDimZ
                       block_x,     // blockDimX
                       block_y,       // blockDimY
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

  gettimeofday(&time_ckernel_f, NULL);


  /**************************** prefix sum *************************************/

  gettimeofday(&time_cscan_s, NULL);

  if(!(presum(&count_dev,(uint)grid_x*block_x*grid_y))){
    printf("presum error\n");
    exit(1);
  }

  gettimeofday(&time_cscan_f, NULL);

  /********************************************************************/


  CUdeviceptr jt_size_dev;

  if(!(jt_size_dev = transport(count_dev, (uint)grid_x*block_x*grid_y, 2, (uint)right))){
    printf("transport error.\n");
    exit(1);
  }

  //getMaxValue(count_dev);
  res = cuMemcpyDtoH(count,jt_size_dev,3 * sizeof(uint)); 
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);    
  }

  uint jt_size = count[1];

  printf("jt_size = %d\tx*b_x*y = %d\tl_p_num = %d\n",jt_size,grid_x*block_x*grid_y,l_p_num);

  gettimeofday(&time_count_f, NULL);

  /***************************************************************************************/





  /************************************************************************
   p memory alloc and p upload
   supplementary:
   The reason that join table is "p" ,it's used sample program .
   As possible I will change appriciate value.
  ************************************************************************/

  gettimeofday(&time_join_s, NULL);


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

  checkCudaErrors(cudaMemset((void *)jt_dev, 0 , jt_size*sizeof(RESULT)));

  /*
  res = cuMemcpyHtoD(jt_dev, jt, jt_size * sizeof(RESULT));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (join) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */

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
                       grid_y,        // gridDimY
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
  res = cuMemFree(p_sum_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (p_sum_dev) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(r_p_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (p_sum_dev) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(l_p_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (p_sum_dev) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(radix_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (p_sum_dev) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  /*size of HtoD data*/

  int DEF = 1000;
  printf("lt = %d\n",left*sizeof(TUPLE)/DEF);
  printf("rt = %d\n" ,right * sizeof(TUPLE)/DEF);
  printf("lL = %d\n", p_num*(left/PER_TH+1)*sizeof(uint)/DEF);
  printf("rL = %d\n", p_num*t_num*sizeof(uint)/DEF);
  printf("plt = %d\n", left*sizeof(TUPLE)/DEF);
  printf("prt = %d\n", right*sizeof(TUPLE)/DEF);
  printf("count = %d\n", grid_x*block_x*block_y*sizeof(uint)/DEF);
  printf("l_p = %d\n", (l_p_num+1)*sizeof(uint)/DEF);
  printf("radix = %d\n", (l_p_num+1)*sizeof(uint)/DEF);
  printf("r_p = %d\n", (p_num+1)*sizeof(uint)/DEF);
  printf("jt = %d\n",  jt_size*sizeof(RESULT)/DEF);

  printf("\n");
  printf("\n");
  printf("all time:\n");
  printDiff(begin, end);
  printf("\n");
  printf("left and right table upload time:\n");
  printDiff(time_upload_s,time_upload_f);
  printf("\n");
  printf("hash time:\n");
  printDiff(time_hash_s,time_hash_f);
  printf("\n");
  printf("lhash time:\n");
  printDiff(time_lhash_s,time_lhash_f);
  printf("lhash count kernel time:\n");
  printDiff(time_lhck_s,time_lhck_f);
  printf("lhash scan time:\n");
  printDiff(time_lscan_s,time_lscan_f);
  printf("lhash kernel time:\n");
  printDiff(time_hkernel_s,time_hkernel_f);
  printf("\n");
  printf("rhash time:\n");
  printDiff(time_rhash_s,time_rhash_f);
  printf("rhash count kernel time:\n");
  printDiff(time_rhck_s,time_rhck_f);
  printf("rhash scan time:\n");
  printDiff(time_rscan_s,time_rscan_f);
  printf("rhash kernel time:\n");
  printDiff(time_rhk_s,time_rhk_f);
  printf("\n");
  printf("count time:\n");
  printDiff(time_count_s,time_count_f);
  printf("count scan time:\n");
  printDiff(time_cscan_s,time_cscan_f);
  printf("count kernel time:\n");
  printDiff(time_ckernel_s,time_ckernel_f);
  printf("\n");
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


  //free(lL);
  //free(rL);
  free(p_sum);
  free(l_p);
  free(radix_num);
  free(p_loc);



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
