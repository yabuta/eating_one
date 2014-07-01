#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
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

static uint iDivUp(uint dividend, uint divisor)
{
    return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
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
  uint *used;//usedなnumberをstoreする
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  for(uint i=0; i<SELECTIVITY ;i++){
    used[i] = i;
  }
  uint selec = SELECTIVITY;

  //uniqueなnumberをvalにassignする
  for (uint i = 0; i < right; i++) {
    if(&(rt[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }
    rt[i].key = getTupleId();
    uint temp = rand()%selec;
    uint temp2 = used[temp];
    selec = selec-1;
    used[temp] = used[selec];

    rt[i].val = temp2; 
  }


  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て*******************************
  res = cuMemHostAlloc((void**)&lt,left * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  uint counter = 0;//matchするtupleをcountする。
  uint *used_r;
  used_r = (uint *)calloc(right,sizeof(uint));
  for(uint i=0; i<right ; i++){
    used_r[i] = i;
  }
  uint rg = right;
  uint l_diff;//
  if(MATCH_RATE != 0){
    l_diff = left/(MATCH_RATE*right);
  }else{
    l_diff = 1;
  }
  for (uint i = 0; i < left; i++) {
    lt[i].key = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*right){
      uint temp = rand()%rg;
      uint temp2 = used_r[temp];
      rg = rg-1;
      used[temp] = used[rg];
      
      lt[i].val = rt[temp2].val;      
      counter++;
    }else{
      uint temp = rand()%selec;
      uint temp2 = used[temp];
      selec = selec-1;
      used[temp] = used[selec];
      lt[i].val = temp2; 
    }
  }

  //printf("%d\n",counter);

  
  free(used);
  free(used_r);
  
  /*********************************************************************************/


  //JOIN_TUPLEへのGPUでも参照できるメモリの割り当て********************************
  res = cuMemHostAlloc((void**)&jt, JT_SIZE * sizeof(RESULT),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to JOIN_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  /**********************************************************************************/


}


/*      memory free           */
void freeTuple(){

  cuMemFreeHost(rt);
  cuMemFreeHost(lt);
  //cuMemFreeHost(jt);

}


void
join()
{

  //TUPLE *plt,*prt;
  RESULT result;
  int resultVal = 0;
  uint jt_size;
  uint p_num,t_num;
  uint r_p_max;
  uint count_size;
  uint *l_p,*radix_num,*p_loc;
  uint *count,*p_sum;
  uint table_type;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function,lcp_function,lp_function,sp_function,rcp_function,rp_function;
  CUmodule module,c_module,p_module,cp_module;
  CUdeviceptr lt_dev, rt_dev, jt_dev,count_dev,lL_dev,rL_dev;
  CUdeviceptr plt_dev,prt_dev,l_p_dev,r_p_dev,radix_dev,p_sum_dev;
  unsigned int block_x, block_y, grid_x, grid_y,p_grid_x,p_block_x;
  char fname[256];
  const char *path=".";
  struct timeval begin, end;
  struct timeval time_alloc_s,time_alloc_f;
  struct timeval time_jdown_s,time_jdown_f,time_lupload_s,time_lupload_f;
  struct timeval time_hash_s,time_hash_f,time_lhck_s,time_lhck_f,time_lhash_s,time_lhash_f,time_lscan_s,time_lscan_f,time_hkernel_s,time_hkernel_f,time_lhother_s,time_lhother_f;
  struct timeval time_rhash_s,time_rhash_f,time_rhck_s,time_rhck_f,time_rscan_s,time_rscan_f,time_rhk_s,time_rhk_f,time_rhother_s,time_rhother_f;
  struct timeval time_cscan_s,time_cscan_f,time_count_s,time_count_f,time_ckernel_s,time_ckernel_f;
  struct timeval time_join_s,time_join_f,time_jkernel_s,time_jkernel_f;
  struct timeval temp_s,temp_f,time_start,time_stop;
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

  cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);


  /*********************************************************************************/



  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *今回はjoin_gpu.cubinとcountJoinTuple.cubinの二つの関数を実行する
   */

  sprintf(fname, "%s/partitioning.cubin", path);
  res = cuModuleLoad(&p_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(partitioning) failed\n");
    exit(1);
  }

  res = cuModuleGetFunction(&lcp_function, p_module, "lcount_partitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(lcount_partitioning) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&lp_function, p_module, "lpartitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(lpartitioning) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&rcp_function, p_module, "rcount_partitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(rcount_partitioning) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&rp_function, p_module, "rpartitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(rpartitioning) failed\n");
    exit(1);
  }  

  res = cuModuleGetFunction(&sp_function, p_module, "countPartition");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(countpartition) failed\n");
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

  /**********************************************************************************/

  gettimeofday(&time_alloc_f, NULL);


  /********************** upload lt , rt , count ,plt, prt, rL, lL***********************/


  gettimeofday(&time_lupload_s, NULL);  

  res = cuMemcpyHtoD(lt_dev, lt, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (lt) failed: res = %lu\n", res);//conv(res));
    exit(1);
  }

  gettimeofday(&time_lupload_f, NULL);  

  /***************************************************************************/

  /****************************************************************
    left table partitioning for hash

  ***************************************************************/

  /*preparation of partitioning*/


  /*
    big table is partition standard and small table is stored in shared memory.
    here , right table is big table.
   */



  p_num=256*1024;
  //p_num=PARTITION*PARTITION*PARTITION;
  t_num = left/LEFT_PER_TH;
  if(left%LEFT_PER_TH != 0){
    t_num++;
  }

  //printf("%d\n",p_num * t_num);

  /*lL,plt and prt alloc in GPU */
  res = cuMemAlloc(&lL_dev, t_num * PARTITION * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lL) failed\n");
    exit(1);
  }
  res = cuMemAlloc(&plt_dev, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (plt) failed\n");
    exit(1);
  }

  printf("t_num=%d\tp_num=%d\n",t_num,p_num);

  p_block_x = t_num < PART_C_NUM ? t_num : PART_C_NUM;
  p_grid_x = t_num / p_block_x;
  if (t_num % p_block_x != 0)
    p_grid_x++;


  CUdeviceptr ltemp; 
  int p_n=0;

  gettimeofday(&time_lhash_s, NULL);


  for(uint loop=0 ; pow(PARTITION,loop)<p_num ; loop++){

    if(p_num<pow(PARTITION,loop+1)){
      p_n = p_num/pow(PARTITION,loop);
    }else{
      p_n = PARTITION;
    }

    gettimeofday(&time_lhck_s, NULL);


    printf("p_grid=%d\tp_block=%d\n",p_grid_x,p_block_x);

    void *count_lpartition_args[]={
    
      (void *)&lt_dev,
      (void *)&lL_dev,
      (void *)&p_n,
      (void *)&t_num,
      (void *)&left,
      (void *)&loop
      
    };
    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         lcp_function,    // CUfunction f
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
    if(res != CUDA_SUCCESS){
      printf("cuLaunchKernel(lhash count) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }      
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS){
      printf("cuCtxSynchronize(lhash count) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  

    gettimeofday(&time_lhck_f, NULL);


    /**************************** prefix sum *************************************/

    gettimeofday(&time_lscan_s, NULL);

    if(!(presum(&lL_dev,t_num*p_n))){
      printf("lL presum error\n");
      exit(1);
    }

    gettimeofday(&time_lscan_f, NULL);

    /********************************************************************/
    gettimeofday(&time_hkernel_s, NULL);
    void *lpartition_args[]={
    
      (void *)&lt_dev,
      (void *)&plt_dev,
      (void *)&lL_dev,
      (void *)&p_n,
      (void *)&t_num,
      (void *)&left,
      (void *)&loop
    };
    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         lp_function,    // CUfunction f
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
      printf("cuLaunchKernel(lhash partition) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }      
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(lhash partition) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    } 
    gettimeofday(&time_hkernel_f, NULL);  

    printf("...loop finish\n");

    ltemp = lt_dev;
    lt_dev = plt_dev;
    plt_dev = ltemp;
  
  }

  gettimeofday(&time_lhash_f, NULL);
  
  p_block_x = 256;
  p_grid_x = left/p_block_x;
  if(left%p_block_x!=0){
    p_block_x++;
  }


  gettimeofday(&time_lhother_s, NULL);  

  CUdeviceptr lstartPos_dev;
  int lpos_size = MAX_LARGE_ARRAY_SIZE*iDivUp(p_num,MAX_LARGE_ARRAY_SIZE);

  res = cuMemAlloc(&lstartPos_dev, lpos_size * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (startPos) failed\n");
    exit(1);
  }
  checkCudaErrors(cudaMemset((void *)lstartPos_dev,0,lpos_size*sizeof(uint)));

  void *lspartition_args[]={
    
    (void *)&lt_dev,
    (void *)&lstartPos_dev,
    (void *)&p_num,
    (void *)&left,
  };
  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
  res = cuLaunchKernel(
                       sp_function,    // CUfunction f
                       p_grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       p_block_x,       // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       lspartition_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS) {
    printf("cuLaunchKernel(lhash partition) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }      
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(lhash partition) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  } 

  p_sum = (uint *)calloc(p_num,sizeof(uint));

  res = cuMemcpyDtoH(p_sum,lstartPos_dev,p_num * sizeof(uint)); 
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (p_sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);    
  }
  
  if(!(presum(&lstartPos_dev,lpos_size))){
    printf("lstartpos presum error\n");
    exit(1);
  }

  gettimeofday(&time_lhother_f, NULL);  




  printf("lhash count kernel time:\n");
  printDiff(time_lhck_s,time_lhck_f);
  printf("lhash scan time:\n");
  printDiff(time_lscan_s,time_lscan_f);
  printf("lhash kernel time:\n");
  printDiff(time_hkernel_s,time_hkernel_f);
  printf("lhash other time:\n");
  printDiff(time_lhother_s,time_lhother_f);



  /***************************
   end of resizing partition

  ***********************************/

  /**************************************************************/


  res = cuMemFree(plt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(lL_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemFree(lt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  gettimeofday(&end, NULL);



  printf("\n");
  printf("all time:\n");
  printDiff(begin, end);
  printf("\n");
  printf("table upload time:\n");
  printDiff(time_lupload_s,time_lupload_f);
  printf("\n");
  printf("hash time:\n");
  printDiff(time_lhash_s,time_lhash_f);
  printf("\n");
  //finish GPU   ****************************************************
  res = cuModuleUnload(p_module);
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
