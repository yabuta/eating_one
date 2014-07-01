#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <math.h> 
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

  printf("%d\n",counter);

  
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
  cuMemFreeHost(jt);

}


void
join()
{

  //TUPLE *plt,*prt;
  RESULT result;
  int resultVal = 0;
  //int loop = LOOP;
  uint fpart=pow(PARTITION,LOOP);
  uint jt_size;
  uint p_num,t_num;
  uint r_p_max;
  uint count_size;
  uint *l_p,*radix_num,*p_loc,*fp_num;
  uint *count;
  uint table_type;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function,cp_function,p1_function,p2_function,pf_function;
  CUmodule module,c_module,p_module,cp_module;
  CUdeviceptr lt_dev, rt_dev, jt_dev,count_dev,blockCount_dev,localScan_dev,startPos_dev;
  CUdeviceptr plt_dev,prt_dev,l_p_dev,r_p_dev,radix_dev;
  unsigned int block_x, block_y, grid_x, grid_y,p_grid_x,p_block_x;
  char fname[256];
  const char *path=".";
  struct timeval begin, end;
  struct timeval time_join_s,time_join_f,time_jkernel_s,time_jkernel_f;
  struct timeval time_jdown_s,time_jdown_f,time_upload_s,time_upload_f;
  struct timeval time_hash_s,time_hash_f,time_hkernel_s,time_hkernel_f,time_lhash_s,time_lhash_f,time_rhash_s,time_rhash_f,time_rscan_s,time_rscan_f,time_cscan_s,time_cscan_f;
  struct timeval time_rhck_s,time_rhck_f,time_rhk_s,time_rhk_f;
  struct timeval time_count_s,time_count_f,time_ckernel_s,time_ckernel_f,time_alloc_s,time_alloc_f;
  struct timeval temp_s,temp_f,time_start,time_stop;
  double time_cal;
  long temper=0,tempest=0;
  long uptime = 0;



  /***********************************************************************************/
  /***********************************************************************************/
  /******************** GPU init here ************************************************/
  /***********************************************************************************/
  /***********************************************************************************/



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
  res = cuModuleGetFunction(&p1_function, p_module, "partitioning1");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(partitioning1) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&p2_function, p_module, "partitioning2");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(partitioning2) failed\n");
    exit(1);
  }
  /*
  res = cuModuleGetFunction(&pf_function, p_module, "partitioningF");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(partitioningF) failed\n");
    exit(1);
  }
  */
  /***********************************************************************************/
  /***********************************************************************************/
  /******************** GPU init finish **********************************************/
  /***********************************************************************************/
  /***********************************************************************************/


  /***********************************************************************************/
  /***********************************************************************************/
  /******************** table create *************************************************/
  /***********************************************************************************/
  /***********************************************************************************/

  createTuple();

  /***********************************************************************************/
  /***********************************************************************************/
  /******************** table create finish ******************************************/
  /***********************************************************************************/
  /***********************************************************************************/


  /***********************************************************************************/
  /***********************************************************************************/
  /******************** hash join part ***********************************************/
  /***********************************************************************************/
  /***********************************************************************************/


  /*******************
   send data:
   TUPLE *rt,*prt
   TUPLE *lt,*plt
   RESULT *jt
   int blockCount
   int localScan

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

  /*lL,plt and prt alloc in GPU */
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

  /**********************************************************************************/



  gettimeofday(&time_alloc_f, NULL);


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

  /***************************************************************************/


  /****************************************************************
    left table partitioning for hash

  ***************************************************************/
  table_type = LEFT;

  gettimeofday(&time_lhash_s, NULL);

  p_block_x = PART_X;
  p_grid_x = left / ONE_BL_NUM;
  if (left % ONE_BL_NUM != 0)
    p_grid_x++;

  printf("p_block_x = %d\tp_grid_x = %d\n",p_block_x,p_grid_x);

  res = cuMemAlloc(&blockCount_dev, PARTITION * p_grid_x * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (blockCount) failed\n");
    exit(1);
  }
  res = cuMemAlloc(&localScan_dev, PARTITION * p_grid_x * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (localScan) failed\n");
    exit(1);
  }
  res = cuMemAlloc(&startPos_dev, fpart * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (localScan) failed\n");
    exit(1);
  }

  checkCudaErrors(cudaMemset((void *)startPos_dev, 0 , fpart *sizeof(uint)));

  struct timeval cnt_s,cnt_f,p1_s,p1_f,p2_s,p2_f;
  long cnt=0,p1=0,p2=0;

  gettimeofday(&time_hash_s, NULL);
  
  for(uint loop=0 ; loop<LOOP ; loop++){


    gettimeofday(&cnt_s, NULL);

    void *count_lpartition_args[]={
    
      (void *)&lt_dev,
      (void *)&blockCount_dev,
      (void *)&localScan_dev,
      (void *)&loop,
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
      printf("count cuLaunchKernel(lhash count) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }      
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(lhash count) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    } 

    gettimeofday(&cnt_f, NULL);
    cnt += (cnt_f.tv_sec - cnt_s.tv_sec) * 1000 * 1000 + (cnt_f.tv_usec - cnt_s.tv_usec);

    /**************************** prefix sum *************************************/
    if(!(presum(&blockCount_dev,PARTITION*p_grid_x))){
      printf("left blockCount presum error\n");
      exit(1);
    }
    /********************************************************************/
    /*
    uint *bc = (uint *)calloc(PARTITION*p_grid_x,sizeof(uint));
    res = cuMemcpyDtoH(bc,blockCount_dev,PARTITION * p_grid_x * sizeof(uint)); 
    if(res != CUDA_SUCCESS){
      printf("cuMemcpyDtoH (bc) failed: res = %lu\n", (unsigned long)res);
      exit(1);    
    }
    
    for(uint j=0; j<PARTITION*p_grid_x; j++){
      printf("%d = %d\n",j,bc[j]);
    }
    */
    //exit(1);

    gettimeofday(&p1_s, NULL);
    void *lpartition1_args[]={    
      (void *)&lt_dev,
      (void *)&plt_dev,
      (void *)&localScan_dev,
      (void *)&loop,
      (void *)&left
    };
    res = cuLaunchKernel(
                         p1_function,    // CUfunction f
                         p_grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         p_block_x,       // blockDimX
                         1,       // blockDimY
                         1,             // blockDimZ
                         0,             // sharedMemBytes
                         NULL,          // hStream
                         lpartition1_args,   // keunelParams
                         NULL           // extra
                         );
    if(res != CUDA_SUCCESS) {
      printf("cuLaunchKernel(lhash partition1) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(lhash partition1) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    } 
    gettimeofday(&p1_f, NULL);
    p1 += (p1_f.tv_sec - p1_s.tv_sec) * 1000 * 1000 + (p1_f.tv_usec - p1_s.tv_usec);

    gettimeofday(&p2_s, NULL);
    void *lpartition2_args[]={
    
      (void *)&lt_dev,
      (void *)&plt_dev,
      (void *)&blockCount_dev,
      (void *)&loop,
      (void *)&left
    };
    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         p2_function,    // CUfunction f
                         p_grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         p_block_x,       // blockDimX
                         1,       // blockDimY
                         1,             // blockDimZ
                         0,             // sharedMemBytes
                         NULL,          // hStream
                         lpartition2_args,   // keunelParams
                         NULL           // extra
                         );
    if(res != CUDA_SUCCESS) {
      printf("cuLaunchKernel(lhash partition2) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }      
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(lhash partition2) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  
    gettimeofday(&p2_f, NULL);
    p2 += (p2_f.tv_sec - p2_s.tv_sec) * 1000 * 1000 + (p2_f.tv_usec - p2_s.tv_usec);

  }

  gettimeofday(&time_lhash_f, NULL);

  printf("lhash time:\n");
  printDiff(time_lhash_s,time_lhash_f);

  printf("count: %ld us (%ld ms)\n", cnt, cnt/1000);
  printf("p1: %ld us (%ld ms)\n", p1, p1/1000);
  printf("p2: %ld us (%ld ms)\n", p2, p2/1000);

  res = cuMemcpyDtoH(lt,lt_dev,left * sizeof(TUPLE)); 
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (lt) failed: res = %lu\n", (unsigned long)res);
    exit(1);    
  }

  int pa=0;
  for(uint i=0 ; i<left ; i++){
    int x = lt[i].val%fpart;
    if(x<pa){
      printf("partition error. x = %d pa = %d i = %d\n",x,pa,i);
      break;
      //exit(1);
    }else{
      pa = x;
    }
  }
  exit(1);

  for(uint i=0 ; i<100 ; i++){
    printf("lt[%d] = %d\t%d\t%d\t%d\n",i,lt[i].val,lt[i].val%fpart,lt[i].val%PARTITION,(lt[i].val>>RADIX)*PARTITION);
  }
  printf("partition success.\npa = %d\nfpart = %d\n",pa,fpart);

  exit(1);

  fp_num = (uint *)calloc(fpart,sizeof(uint));

  res = cuMemcpyDtoH(fp_num,startPos_dev,fpart * sizeof(uint)); 
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (p_sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);    
  }

  /***********************************************
    resizing partition 

  ************************************************/
  uint l_p_num = 0;
  for(uint i=0 ; i<fpart ;i++ ){
    if(fp_num[i]%B_ROW_NUM == 0 && fp_num[i]!=0){
      l_p_num += fp_num[i]/B_ROW_NUM;
    }else{
      l_p_num += fp_num[i]/B_ROW_NUM + 1;
    }            
  }
  l_p = (uint *)calloc(l_p_num+1,sizeof(uint));
  radix_num = (uint *)calloc(l_p_num+1,sizeof(uint));
  p_loc = (uint *)calloc(fpart,sizeof(uint));
  l_p_num = 0;
  uint temp2 = 0;

  /****************presum*****************/
  if(!(presum(&startPos_dev,fpart))){
    printf("left blockCount presum error\n");
    exit(1);
  }
  res = cuMemcpyDtoH(p_loc,startPos_dev,fpart * sizeof(uint)); 
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (p_sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);    
  }
  /***************************************/

  for(int i=0; i<fpart; i++){
    if(fp_num[i]/B_ROW_NUM < 1 || fp_num[i]==B_ROW_NUM){
      l_p[l_p_num] = p_loc[i];
      radix_num[l_p_num] = i;
      l_p_num++;
    }else{
      if(fp_num[i]%B_ROW_NUM == 0){
        temp2 = fp_num[i]/B_ROW_NUM;
      }else{
        temp2 = fp_num[i]/B_ROW_NUM + 1;
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
  radix_num[l_p_num] = fpart;
  /***************************
   end of resizing partition

  ***********************************/

  res = cuMemFree(plt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (plt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  /*
  res = cuMemFree(p_sum_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (p_sum_dev) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */

  /**************************************************************/

  gettimeofday(&time_lhash_f, NULL);


  /****************************************************************
    right table partitioning for hash


  ***************************************************************/

  //  table_type = RIGHT;

  gettimeofday(&time_rhash_s, NULL);

  p_block_x = PART_X;
  p_grid_x = right / ONE_BL_NUM;
  if (right % ONE_BL_NUM != 0)
    p_grid_x++;

  printf("p_block_x = %d\tp_grid_x = %d\n",p_block_x,p_grid_x);

  checkCudaErrors(cudaMemset((void *)startPos_dev, 0 , fpart*sizeof(uint)));

  for(uint r_loop=0 ; r_loop<LOOP ; r_loop++){
    checkCudaErrors(cudaMemset((void *)blockCount_dev, 0 , PARTITION*p_grid_x*sizeof(uint)));
    checkCudaErrors(cudaMemset((void *)localScan_dev, 0 , PARTITION*p_grid_x*sizeof(uint)));

    gettimeofday(&time_rhck_s, NULL);

    void *count_rpartition_args[]={
    
      (void *)&rt_dev,
      (void *)&blockCount_dev,
      (void *)&localScan_dev,
      (void *)&r_loop,
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
      printf("cuLaunchKernel(rhash count) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(rhash count) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  


    gettimeofday(&time_rhck_f, NULL);


    /**************************** prefix sum *************************************/

    gettimeofday(&time_rscan_s, NULL);
    if(!(presum(&blockCount_dev,PARTITION*p_grid_x))){
      printf("presum error\n");
      exit(1);
    }
    gettimeofday(&time_rscan_f, NULL);

    /********************************************************************/

    /*
    void *rpartition1_args[]={
    
      (void *)&rt_dev,
      (void *)&prt_dev,
      (void *)&localScan_dev,
      (void *)&r_loop,
      (void *)&right
    };

    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

    res = cuLaunchKernel(
                         p1_function,    // CUfunction f
                         p_grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         p_block_x,       // blockDimX
                         1,       // blockDimY
                         1,             // blockDimZ
                         0,             // sharedMemBytes
                         NULL,          // hStream
                         rpartition1_args,   // keunelParams
                         NULL           // extra
                         );
    if(res != CUDA_SUCCESS) {
      printf("cuLaunchKernel(rhash partition1) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }      

    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(rhash partition1) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  
    */

    void *rpartition2_args[]={
    
      (void *)&rt_dev,
      (void *)&prt_dev,
      (void *)&blockCount_dev,
      (void *)&r_loop,
      (void *)&right
    };

    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

    res = cuLaunchKernel(
                         p2_function,    // CUfunction f
                         p_grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         p_block_x,       // blockDimX
                         1,       // blockDimY
                         1,             // blockDimZ
                         0,             // sharedMemBytes
                         NULL,          // hStream
                         rpartition2_args,   // keunelParams
                         NULL           // extra
                         );
    if(res != CUDA_SUCCESS) {
      printf("cuLaunchKernel(rhash partition2) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }      

    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(rhash partition2) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  

    /*
    if(loop!=LOOP){
      void *rpartition2_args[]={
    
        (void *)&rt_dev,
        (void *)&prt_dev,
        (void *)&blockCount_dev,
        (void *)&r_loop,
        (void *)&right
      };

      //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

      res = cuLaunchKernel(
                           p2_function,    // CUfunction f
                           p_grid_x,        // gridDimX
                           1,        // gridDimY
                           1,             // gridDimZ
                           p_block_x,       // blockDimX
                           1,       // blockDimY
                           1,             // blockDimZ
                           0,             // sharedMemBytes
                           NULL,          // hStream
                           rpartition2_args,   // keunelParams
                           NULL           // extra
                           );
      if(res != CUDA_SUCCESS) {
        printf("cuLaunchKernel(rhash partition2) failed: res = %lu\n", (unsigned long int)res);
        exit(1);
      }      

      res = cuCtxSynchronize();
      if(res != CUDA_SUCCESS) {
        printf("cuCtxSynchronize(rhash partition2) failed: res = %lu\n", (unsigned long int)res);
        exit(1);
      }  
      
    }else{
      void *rpartitionF_args[]={
        
        (void *)&lt_dev,
        (void *)&plt_dev,
        (void *)&blockCount_dev,
        (void *)&startPos_dev,        
        (void *)&r_loop,
        (void *)&right
      };
      //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
      res = cuLaunchKernel(
                           pf_function,    // CUfunction f
                           p_grid_x,        // gridDimX
                           1,        // gridDimY
                           1,             // gridDimZ
                           p_block_x,       // blockDimX
                           1,       // blockDimY
                           1,             // blockDimZ
                           0,             // sharedMemBytes
                           NULL,          // hStream
                           rpartitionF_args,   // keunelParams
                           NULL           // extra
                           );
      if(res != CUDA_SUCCESS) {
        printf("cuLaunchKernel(lhash partitionF) failed: res = %lu\n", (unsigned long int)res);
        exit(1);
      }      
      res = cuCtxSynchronize();
      if(res != CUDA_SUCCESS) {
        printf("cuCtxSynchronize(lhash partitionF) failed: res = %lu\n", (unsigned long int)res);
        exit(1);
      }  
    }
    */
  }

  /*
  res = cuMemcpyDtoH(rt,prt_dev,right * sizeof(TUPLE)); 
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (p_sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);    
  }

  int pa=0;
  for(uint i=0 ; i<right ; i++){
    int x = rt[i].val%fpart;
    if(x<pa){
      printf("partition error. x = %d pa = %d i = %d\n",x,pa,i);
      exit(1);
    }else{
      pa = x;
    }
  }

  printf("partition success.\nfpart = %d\n",fpart);
  exit(1);
  */

  uint *r_p =  (uint *)calloc(fpart+1,sizeof(uint));
  uint rdiff;
  if(!(presum(&startPos_dev,fpart))){
    printf("left blockCount presum error\n");
    exit(1);
  }
  res = cuMemcpyDtoH(r_p,startPos_dev,fpart * sizeof(uint)); 
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (r_p) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  r_p[fpart] = right;
  for(uint i = 0; i<fpart+1 ;i++){
    if(i==0){
      r_p_max = r_p[i];
    }else{
      rdiff = r_p[i] - r_p[i-1];
      if(rdiff > r_p_max){
        r_p_max = rdiff;
      }
    }
  }

  res = cuMemFree(prt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (prt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(blockCount_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (blockCount) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(localScan_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (localScan) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(startPos_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (startPos) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


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

  p_num = fpart;

  gettimeofday(&time_count_s, NULL);


  block_x = r_p_max < BLOCK_SIZE_X ? r_p_max : BLOCK_SIZE_X;
  block_y = BLOCK_SIZE_Y;
  grid_x = l_p_num;
  grid_y = GRID_SIZE_Y;

  count_size = grid_x * grid_y * block_x + 1;

  printf("r_p_max = %d\n",r_p_max);
  printf("block_x = %d\tgrid_x = %d\tcount_size = %d\n",block_x,grid_x,count_size);


  /*GPU memory alloc and send data of count ,l_p ,radix and r_p*/

  res = cuMemAlloc(&count_dev, count_size * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }
  res = cuMemAlloc(&l_p_dev, (l_p_num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (l_p) failed\n");
    exit(1);
  }
  res = cuMemAlloc(&r_p_dev, (l_p_num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (r_p) failed\n");
    exit(1);
  }
  res = cuMemAlloc(&radix_dev, (l_p_num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (radix) failed\n");
    exit(1);
  }

  //checkCudaErrors(cudaMemset((void *)count_dev, 0 , count_size*sizeof(uint)));

  res = cuMemcpyHtoD(l_p_dev, l_p, (l_p_num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (l_p_dev) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemcpyHtoD(r_p_dev, r_p, (l_p_num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (r_p_dev) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemcpyHtoD(radix_dev, radix_num, (l_p_num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (radix) failed: res = %lu\n", (unsigned long)res);
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
    
    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&count_dev,
    (void *)&r_p_dev,
    (void *)&radix_dev,
    (void *)&l_p_dev,
    (void *)&right,
    (void *)&left
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       c_function,    // CUfunction f
                       grid_x,        // gridDimX
                       grid_y,        // gridDimY
                       1,             // gridDimZ
                       block_x,     // blockDimX
                       block_y,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       count_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS) {
    printf("cuLaunchKernel(count) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }      
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  

  gettimeofday(&time_ckernel_f, NULL);


  /**************************** prefix sum *************************************/

  gettimeofday(&time_cscan_s, NULL);

  if(!(presum(&count_dev,count_size))){
    printf("presum error\n");
    exit(1);
  }

  gettimeofday(&time_cscan_f, NULL);

  /********************************************************************/

  if(!getValue(count_dev,(uint)count_size,&jt_size)){
    printf("getValue(count_dev) error.\n");
    exit(1);
  }

  printf("jt_size = %d\tx*b_x*y = %d\tl_p_num = %d\n",jt_size,grid_x*block_x*grid_y,l_p_num);

  gettimeofday(&time_count_f, NULL);

  /***************************************************************************************/


  /************************************************************************
   p memory alloc and p upload
  ************************************************************************/

  gettimeofday(&time_join_s, NULL);


  if(jt_size <= 0){
    printf("no tuple is matched.\n");
  }else{
    res = cuMemAlloc(&jt_dev, jt_size * sizeof(RESULT));
    if (res != CUDA_SUCCESS) {
      printf("cuMemAlloc (jt) failed\n");
      exit(1);
    }

    //jt = (RESULT *)malloc(jt_size*sizeof(RESULT)); 
  
    gettimeofday(&time_jkernel_s, NULL);

    void *kernel_args[]={
      (void *)&lt_dev,
      (void *)&rt_dev,
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
      printf("cuLaunchKernel(join) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  



    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(join) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  

    gettimeofday(&time_jkernel_f, NULL);

    gettimeofday(&time_jdown_s, NULL);

    res = cuMemcpyDtoH(jt, jt_dev, jt_size * sizeof(RESULT));
    if (res != CUDA_SUCCESS) {
      printf("cuMemcpyDtoH (jt) failed: res = %lu\n", (unsigned long)res);
      exit(1);
    }
    printf("jt_size = %d\n",jt_size);

    gettimeofday(&time_jdown_f, NULL);


    /********************************************************************/

    gettimeofday(&time_join_f, NULL);
  }

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

  /***********************************************************************************/
  /***********************************************************************************/
  /******************** hash join part finish*****************************************/
  /***********************************************************************************/
  /***********************************************************************************/


  /*size of HtoD data*/

  int DEF = 1000;
  printf("lt = %d\n",left*sizeof(TUPLE)/DEF);
  printf("rt = %d\n" ,right * sizeof(TUPLE)/DEF);
  printf("plt = %d\n", left*sizeof(TUPLE)/DEF);
  printf("prt = %d\n", right*sizeof(TUPLE)/DEF);
  printf("l_p = %d\n", (l_p_num+1)*sizeof(uint)/DEF);
  printf("radix = %d\n", (l_p_num+1)*sizeof(uint)/DEF);
  printf("r_p = %d\n", (p_num+1)*sizeof(uint)/DEF);
  printf("count = %d\n", grid_x*block_x*block_y*sizeof(uint)/DEF);
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
  printf("\n");
  printf("rhash time:\n");
  printDiff(time_rhash_s,time_rhash_f);
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

  free(fp_num);
  free(l_p);
  free(radix_num);
  free(p_loc);
  free(r_p);

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
