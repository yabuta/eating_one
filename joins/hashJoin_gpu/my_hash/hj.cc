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
#include "debug.h"
#include "tuple.h"

TUPLE *rt;
TUPLE *lt;
RESULT *jt;

TUPLE *hrt;
TUPLE *hlt;

int rlocation[NB_BKTENT+1];
int rcount[NB_BKTENT];
int llocation[NB_BKTENT+1];
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



//初期化する
void
init(void)
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
    //gettimeofday(&(Tright[i].t), NULL);
    rt[i].id = getTupleId();

    for(int j = 0;j<VAL_NUM;j++){
      rt[i].val[j] = rand()%SELECTIVITY; // selectivity for jt size = 1000000 
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
    //gettimeofday(&(Tleft[i].t), NULL);    
    lt[i].id = getTupleId();

    for(int j = 0; j < VAL_NUM;j++){
      lt[i].val[j] = rand()%SELECTIVITY; // selectivity for jt size = 1000000 
    }
    
  }
  
  /*********************************************************************************/

  //JOIN_TUPLEへのGPUでも参照できるメモリの割り当て********************************
  res = cuMemHostAlloc((void**)&jt,JT_SIZE * sizeof(RESULT),CU_MEMHOSTALLOC_DEVICEMAP);
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
    Tjoin[i].id = getTupleId();
  }

  /**********************************************************************************/

  
}

void freeTuple(){


  cuMemFreeHost(jt);
  cuMemFreeHost(rt);
  cuMemFreeHost(lt);

}


// create index for S
void
createPart()
{

  /*
  int fd;
  int *bfdAry;
  char partFile[BUFSIZ];
  TUPLE buf[NB_BUF];
  */

  for(int i = 0; i<right ; i++){
    int idx = rt[i].val % NB_BKTENT;
    rcount[idx]++;
  }

  rlocation[0] = 0;
  for(int i = 1; i<NB_BKTENT ; i++){
    rlocation[i] = rlocation[i-1] + rcount[i-1];
  }
  rlocation[NB_BKTENT] = right;

  //for count[] reuse
  for(int i = 0; i<NB_BKTENT ; i++){
    rcount[i] = 0;
  }

  for(int i = 0; i<right ; i++){
    int idx = rt[i].val % NB_BKTENT;
    hrt[rlocation[idx] + rcount[idx]].key = rt[i].key; 
    hrt[rlocation[idx] + rcount[idx]].val = rt[i].val; 
    rcount[idx]++;
  }
  
  for(int i = 0; i<left ; i++){
    int idx = lt[i].val % NB_BKTENT;
    lcount[idx]++;
  }

  llocation[0] = 0;
  for(int i = 1; i<NB_BKTENT ; i++){
    llocation[i] = llocation[i-1] + lcount[i-1];
  }
  llocation[NB_BKTENT] = left;

  //for count[] reuse
  for(int i = 0; i<NB_BKTENT ; i++){
    lcount[i] = 0;
  }

  for(int i = 0; i<left ; i++){
    int idx = lt[i].val % NB_BKTENT;
    hlt[llocation[idx] + lcount[idx]].key = lt[i].key; 
    hlt[llocation[idx] + lcount[idx]].val = lt[i].val; 
    lcount[idx]++;

  }
  
}

int
openPart(const char *partFile, int id)
{
  int fd;
  char buf[BUFSIZ];

  bzero(buf, sizeof(buf));
  sprintf(buf, "hash-part-%s-%d", partFile, id);
  fd = open(buf, O_RDONLY);
  if (fd == -1) ERR;

  return fd;
}

int 
main(int argc,char *argv[])
{

  RESULT result;
  int resultVal = 0;
  struct timeval begin, end;

  int *count;//maybe long long int is better
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function;
  CUmodule module,c_module;
  CUdeviceptr lt_dev, rt_dev, p_dev,count_dev, pre_dev;
  CUdeviceptr ltn_dev, rtn_dev;
  unsigned int block_x, block_y, grid_x, grid_y, grid_z;
  char fname[256];
  const char *path=".";
  struct timeval tv_cal_s, tv_cal_f,time_join_s,time_join_f,time_upload_s,time_upload_f,time_download_s,time_download_f;
  struct timeval time_count_s,time_count_f,time_Cupload_s,time_Cupload_f,time_Cdownload_s,time_Cdownload_f;
  struct timeval time_scan_s,time_scan_f,time_alloc_s,time_alloc_f,time_free_s,time_free_f;
  struct timeval time_Pupload_s,time_Pupload_f,time_send_s,time_send_f;
  double time_cal;


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

  createTuple();


  gettimeofday(&begin, NULL);
  // Hash construction phase
  createPart();


  /*
  for (int i = 0; i < NB_BKTENT; i++) {
    printf("%d\t%d\t%d\t%d\n",rlocation[i],rcount[i],llocation[i],lcount[i]);

  }
  */

  /************** block_x * block_y is decided by BLOCK_SIZE. **************/

  /*注意！
   *LEFTをx軸、RIGHTをy軸にした。ほかの場所と逆になっているので余裕があったら修正する
   *
   */

  int x_size;
  int y_size;

  x_size = rcount[0];
  for(int i=1; i<NB_BKTENT ;i++){
    if(rcount[i-1] < rcount[i]){
      x_size = rcount[i];
    }
  }

  y_size = lcount[0];
  for(int i=1; i<NB_BKTENT ;i++){
    if(lcount[i-1] < lcount[i]){
      y_size = lcount[i];
    }
  }

  block_x = x_size < BLOCK_SIZE_X ? x_size : BLOCK_SIZE_X;
  block_y = y_size < BLOCK_SIZE_Y ? y_size : BLOCK_SIZE_Y;


  grid_x = x_size / block_x;
  if (x_size % block_x != 0)
    grid_x++;

  grid_y = y_size / block_y;
  if (y_size % block_y != 0)
    grid_y++;

  grid_z = NB_BKTENT;

  //malloc memory and 0 for count
  count = (int *)calloc(grid_x * block_y * grid_y,sizeof(int));

  /********************************************************************************/




  /********************************************************************
   *a,b,cのメモリを割り当てる。
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

  
  res = cuMemAlloc(&count_dev, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }

  gettimeofday(&time_alloc_f, NULL);
  /**********************************************************************************/


  
  /********************** upload lt , rt and count***********************/

  /*count uploadの時間計測*/
  gettimeofday(&time_count_s, NULL);

  gettimeofday(&time_send_s, NULL);
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

  res = cuMemcpyHtoD(count_dev, count, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  gettimeofday(&time_send_f, NULL);


  /***************************************************************************/


  /******************************************************************
    count the number of match tuple
    
  *******************************************************************/
  //count.cuの時間計測
  //gettimeofday(&time_count_s, NULL);


  void *count_args[]={
    
    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&count_dev,
    (void *)&arg_left,
    (void *)&arg_right
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  //printf("ok\n");
  res = cuLaunchKernel(
                       c_function,    // CUfunction f
                       grid_x,        // gridDimX
                       grid_y,        // gridDimY
                       grid_z,             // gridDimZ
                       1,       // blockDimX
                       block_y,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       count_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS) {
    printf("cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }      

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  

  //count.cuの時間計測
  //gettimeofday(&time_count_f, NULL);


  //count downloadの時間計測
  //gettimeofday(&time_Cdownload_s, NULL);

  res = cuMemcpyDtoH(count, count_dev, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  //count downloadの時間計測
  //gettimeofday(&time_Cdownload_f, NULL);    

  /***************************************************************************************/



  /**************************** prefix sum *************************************/
  /*  
  for(i=1;i<grid_x*grid_y*block_y;i++){
    count[i] = count[i] + count[i-1];
  }
  */

  thrust::inclusive_scan(count,count+grid_x*grid_y*block_y,count);
  /*
  for(i=1;i<grid_x*grid_y*block_y;i++){
    printf("%d\n",count[i]);
  }
  */
  //exit(1);

  /********************************************************************/



  //cpy count to GPU again      ***************************************
  res = cuMemcpyHtoD(count_dev, count, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  gettimeofday(&time_count_f, NULL);

  /**********************************************************************/


  /************************************************************************
   p memory alloc and p upload
   supplementary:
   The reason that join table is "p" ,it's used sample program .
   As possible I will change appriciate value.
  ************************************************************************/

  if(count[grid_x*grid_y*block_y-1]<=0){
    printf("no tuple is matched.\n");
    exit(1);
  }else{
    res = cuMemAlloc(&p_dev, count[grid_x*grid_y*block_y-1] * sizeof(JOIN_TUPLE));
    if (res != CUDA_SUCCESS) {
      printf("cuMemAlloc (join) failed\n");
      exit(1);
    }
  }

  //p uploadの時間計測
  gettimeofday(&time_Pupload_s, NULL);    

  res = cuMemcpyHtoD(p_dev, Tjoin, count[grid_x*grid_y*block_y-1] * sizeof(JOIN_TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (join) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  //p uploadの時間計測
  gettimeofday(&time_Pupload_f, NULL);    

  //実際のjoinの計算時間
  gettimeofday(&time_join_s, NULL);

  void *kernel_args[]={
    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&p_dev,
    (void *)&count_dev,
    (void *)&arg_left,
    (void *)&arg_right,    
    };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
  res = cuLaunchKernel(
                       function,      // CUfunction f
                       grid_x,        // gridDimX
                       grid_y,        // gridDimY
                       grid_z,             // gridDimZ
                       1,       // blockDimX
                       block_y,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       kernel_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS) {
    printf("cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  



  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  
  

  /*実際のjoinの計算時間*/
  gettimeofday(&time_join_f, NULL);
  //downloadの時間計測
  gettimeofday(&time_download_s, NULL);


  res = cuMemcpyDtoH(Tjoin, p_dev, count[grid_x*grid_y*block_y-1] * sizeof(JOIN_TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (p) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  //downloadの時間計測
  gettimeofday(&time_download_f, NULL);
  

  /***************************************************************/


  //free GPU memory***********************************************


  gettimeofday(&time_free_s, NULL);
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
  
  res = cuMemFree(p_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (jointuple) failed: res = %lu\n", (unsigned long)res);
    exit(1);
    }

  res = cuMemFree(count_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  gettimeofday(&time_free_f, NULL);
  /********************************************************************/

  //finish GPU   ****************************************************
  res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload module failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  
  res = cuModuleUnload(c_module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload c_module failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  
  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  /****************************************************************************/

  //割り当てたメモリを開放する
  tuple_free();



  // Matching phase

  for (int i = 0; i < NB_BKTENT; i++) {

    for (int j = 0; j < rcount[i]; j++) {
      for (int k = 0; k < lcount[i]; k++) {
        if (hrt[rlocation[i] + j].val == hlt[llocation[i] + k].val) {
          result.rkey = hrt[rlocation[i] + j].key;
          result.rval = hrt[rlocation[i] + j].val;
          result.skey = hlt[llocation[i] + k].key;
          result.sval = hlt[llocation[i] + k].val;
          resultVal++;
        }
      }
    }
  }

  gettimeofday(&end, NULL);

  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);

  freeTuple();

  return 0;
}


