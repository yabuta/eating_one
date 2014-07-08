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
#include "debug.h"
#include "scan_common.h"
#include "tuple.h"

BUCKET *Bucket;
int *upper;

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

void shuffle(TUPLE ary[],int size) {    
  for(int i=0;i<size;i++){
    int j = rand()%size;
    int t = ary[i].val;
    ary[i].val = ary[j].val;
    ary[j].val = t;
  }
}

void createTuple()
{

  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て
  CUresult res;

  //メモリ割り当てを行う
  //タプルに初期値を代入


  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て****************************
  res = cuMemHostAlloc((void**)&rt,right * sizeof(TUPLE),CU_MEMHOSTALLOC_PORTABLE);
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
  res = cuMemHostAlloc((void**)&lt,left * sizeof(TUPLE),CU_MEMHOSTALLOC_PORTABLE);
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

  shuffle(lt,left);

  //JOIN_TUPLEへのGPUでも参照できるメモリの割り当て********************************
  res = cuMemHostAlloc((void**)&jt, JT_SIZE * sizeof(RESULT),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to JOIN_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
}


/*      memory free           */
void freeTuple(){

  cuMemFreeHost(rt);
  cuMemFreeHost(lt);
  cuMemFreeHost(jt);
  free(Bucket);

}




void conf(int left,int right,int dep,int wid){

  int m=(left+right)/2;
  int pos = pow(2,dep-1)-1+wid-1;
  upper[pos]=Bucket[m].val;
  wid = (wid-1)*2;
  if(dep==DEPTH-1){
    return;
  }
  conf(left,m-1,dep+1,wid+1);
  conf(m+1,right,dep+1,wid+2);

}

/*
void conf2(int left,int right,int dep,int wid){

  int m=(left+right)/2;
  int pos = pow(2,dep-1)-1+wid-1;
  if(upper[pos]!=Bucket[m].val){
    printf("error\n");
  }
  wid = (wid-1)*2;
  if(dep==13){
    return;
  }
  conf(left,m-1,dep+1,wid+1);
  conf(m+1,right,dep+1,wid+2);

}
*/



/***************create index*****************************/
/*
  swap(TUPLE *a,TUPLE *b)
  qsort(int p,int q)
  createIndex(void)
*/

void swap(BUCKET *a,BUCKET *b)
{
  BUCKET temp;
  temp = *a;
  *a=*b;
  *b=temp;
}

void qsort(int p,int q)
{
  int i,j;
  int pivot;

  i = p;
  j = q;

  pivot = Bucket[(p+q)/2].val;

  while(1){
    while(Bucket[i].val < pivot) i++;
    while(pivot < Bucket[j].val) j--;
    if(i>=j) break;
    swap(&Bucket[i],&Bucket[j]);
    i++;
    j--;
  }
  if(p < i-1) qsort(p,i-1);
  if(j+1 < q) qsort(j+1,q);

}

void createIndex(void)
{

  if (!(Bucket = (BUCKET *)malloc(right * sizeof(BUCKET)))) ERR;
  for(uint i=0; i<right ; i++){
    Bucket[i].val = rt[i].val;
    Bucket[i].adr = i;
  }

  qsort(0,right-1);

  for(uint i=1; i<right ; i++){
    if(Bucket[i-1].val > Bucket[i].val){
      printf("sort error[%d]\n",i);
      break;
    }
  }

}


void join(){

  //uint *count;
  uint jt_size;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function;
  CUmodule module,c_module;
  CUdeviceptr lt_dev, rt_dev, jt_dev, bucket_dev, upper_dev;
  CUdeviceptr ltn_dev, rtn_dev, jt_size_dev;
  CUdeviceptr c_dev,temp_dev;
  unsigned int block_x, grid_x;
  char fname[256];
  const char *path=".";
  struct timeval begin, end;
  struct timeval time_join_s,time_join_f,time_send_s,time_send_f;
  struct timeval time_count_s,time_count_f,time_tsend_s,time_tsend_f,time_isend_s,time_isend_f;
  struct timeval time_jdown_s,time_jdown_f,time_jkernel_s,time_jkernel_f;
  struct timeval time_scan_s,time_scan_f,time_alloc_s,time_alloc_f,time_index_s,time_index_f;
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
  res = cuModuleGetFunction(&c_function, module, "count");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction() failed\n");
    exit(1);
  }

  /*tuple and index init******************************************/  

  printf("create tuple starting...\n");
  createTuple();
  shuffle(lt,left);

  printf("create index starting...\n");
  gettimeofday(&time_index_s, NULL);
  createIndex();
  gettimeofday(&time_index_f, NULL);

  printf("join starting...\n");
  gettimeofday(&begin, NULL);
  /****************************************************************/

  upper = (int *)calloc(SHARED_SIZE,sizeof(int));
  conf(0,right,1,1);

  block_x = left < BLOCK_SIZE_X ? left : BLOCK_SIZE_X;  
  grid_x = left / block_x;
  if (left % block_x != 0)
    grid_x++;
  

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
  /* bucket */
  res = cuMemAlloc(&upper_dev, SHARED_SIZE * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (upper) failed\n");
    exit(1);
  }

  /*count */
  res = cuMemAlloc(&c_dev, (left+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
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
  res = cuMemcpyHtoD(upper_dev, upper, SHARED_SIZE * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (int) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  gettimeofday(&time_isend_f, NULL);
  gettimeofday(&time_send_f, NULL);

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
    (void *)&upper_dev,
    (void *)&right,
    (void *)&left
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       c_function,    // CUfunction f
                       grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       block_x,       // blockDimX
                       1,       // blockDimY
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
    printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  

  /***************************************************************************************/

  /*
  TUPLE temp_a;

  res = cuMemcpyDtoH(&temp_a, lt_dev, sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (temp_a) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  printf("key = %d\n",temp_a.key);
  printf("ok\n");
  exit(1);
  */

  /**************************** prefix sum *************************************/
  gettimeofday(&time_scan_s, NULL);

  if(!(presum(&c_dev,(uint)left+1))){
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

  if(!transport(c_dev,(uint)left+1,&jt_size)){
    printf("transport error.\n");
    exit(1);
  }
  if(jt_size <= 0){
    printf("no tuple is matched.\n");
  }else{
    res = cuMemAlloc(&jt_dev, jt_size * sizeof(RESULT));
    if (res != CUDA_SUCCESS) {
      printf("cuMemAlloc (join) failed\n");
      exit(1);
    }
    //jt = (RESULT *)malloc(jt_size*sizeof(RESULT));
    gettimeofday(&time_jkernel_s, NULL);

    void *kernel_args[]={
      (void *)&rt_dev,
      (void *)&lt_dev,
      (void *)&jt_dev,
      (void *)&c_dev,
      (void *)&bucket_dev,
      (void *)&upper_dev,
      (void *)&right,
      (void *)&left    
    };

    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         function,      // CUfunction f
                         grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         block_x,       // blockDimX
                         1,       // blockDimY
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
      printf("cuCtxSynchronize(join) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  

    gettimeofday(&time_jkernel_f, NULL);

    gettimeofday(&time_join_f, NULL);

    gettimeofday(&time_jdown_s, NULL);
    res = cuMemcpyDtoH(jt, jt_dev, jt_size * sizeof(RESULT));
    if (res != CUDA_SUCCESS) {
      printf("cuMemcpyDtoH (p) failed: res = %lu\n", (unsigned long)res);
      exit(1);
    }
    gettimeofday(&time_jdown_f, NULL);

  }



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
  res = cuMemFree(upper_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (upper) failed: res = %lu\n", (unsigned long)res);
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

  gettimeofday(&end, NULL);


  printf("********index create time**************\n");
  printDiff(time_index_s,time_index_f);

  printf("\n************execution time****************\n\n");
  printf("all time:\n");
  printDiff(begin, end);
  printf("\n");
  printf("gpu memory alloc time:\n");
  printDiff(time_alloc_s,time_alloc_f);
  printf("\n");
  printf("data send time:\n");
  printDiff(time_send_s,time_send_f);
  printf("table data send time:\n");
  printDiff(time_tsend_s,time_tsend_f);
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
  printf("\n");

  for(uint i=0;i<3&&i<jt_size;i++){
    printf("join[%d]:left %8d \t:right: %8d\n",i,jt[i].lkey,jt[i].rkey);
    printf("left = %8d\tright = %8d\n",jt[i].lval,jt[i].rval);
    printf("\n");
  }  

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

  /*
  res = cuModuleUnload(c_module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload module failed: res = %lu\n", (unsigned long)res);
    exit(1);
  } 
  */ 
  
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
