#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "tuple.h"
#include "scan_common.h"


TUPLE *Tright;
TUPLE *Tleft;
JOIN_TUPLE *Tjoin;

//main引数を受け取るグローバル変数
int arg_right;
int arg_left;

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
  res = cuMemHostAlloc((void**)&Tright,arg_right * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to RIGHT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  srand((unsigned)time(NULL));
  uint *used;
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  uint diff;
  if(MATCH_RATE != 0){
    diff = 1/MATCH_RATE;
  }else{
    diff = 1;
  }
  uint counter = 0;

  for (uint i = 0; i < arg_right; i++) {
    if(&(Tright[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }
    Tright[i].id = getTupleId();
    if(i%diff == 0 && counter < arg_right*MATCH_RATE){
      uint temp = rand()%SELECTIVITY;
      while(used[temp] == 1) temp = rand()%SELECTIVITY;
      used[temp] = 1;
      Tright[i].val = temp; // selectivity = 1.0
      counter++;
    }else{
      Tright[i].val = SELECTIVITY + rand()%SELECTIVITY;
    }
  }

  free(used);
  counter = 0;


  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て*******************************
  res = cuMemHostAlloc((void**)&Tleft,arg_left * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  uint l_diff;
  if(MATCH_RATE != 0){
    l_diff = arg_left/(MATCH_RATE*arg_right);
  }else{
    l_diff = 1;
  }
  for (uint i = 0; i < arg_left; i++) {
    Tleft[i].id = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*arg_right){
      Tleft[i].val = Tright[counter*diff].val;
      counter++;
    }else{
      Tleft[i].val = 2 * SELECTIVITY + rand()%SELECTIVITY;
    }
  }

  /*
  for (uint i = 0; i < arg_left; i++) {
    if(&(Tleft[i])==NULL){
      printf("left TUPLE allocate error.\n");
      exit(1);
    }    
    Tleft[i].id = getTupleId();
    if(i%diff == 0 && counter < arg_right*MATCH_RATE){
      Tleft[i].val = Tright[i].val; // selectivity = 1.0
      counter++;
    }else{
      Tleft[i].val = 2 * SELECTIVITY + rand()%SELECTIVITY;
    }
  }
  */

  /*********************************************************************************/

  //JOIN_TUPLEへのGPUでも参照できるメモリの割り当て********************************
  /*
  res = cuMemHostAlloc((void**)&Tjoin,JT_SIZE * sizeof(JOIN_TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to JOIN_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */
  /**********************************************************************************/

  
}


//メモリ解放のため新しく追加した関数。バグがあるかも
void
tuple_free(void){

  cuMemFreeHost(Tright);
  cuMemFreeHost(Tleft);
  cuMemFreeHost(Tjoin);
}



//HrightとHleftをそれぞれ比較する。GPUで並列化するforループもここにあるもので行う。
void
join()
{

  int i, j, idx;
  int *count;//maybe long long int is better
  uint jt_size;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function;
  CUmodule module,c_module;
  CUdeviceptr lt_dev, rt_dev, jt_dev,count_dev, pre_dev;
  CUdeviceptr ltn_dev, rtn_dev;
  unsigned int block_x, block_y, grid_x, grid_y;
  char fname[256];
  const char *path=".";
  struct timeval time_join_s,time_join_f,time_download_s,time_download_f;
  struct timeval time_count_s,time_count_f;
  struct timeval time_scan_s,time_scan_f;
  struct timeval time_send_s,time_send_f;
  struct timeval begin,end;

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
    printf("cuModuleGetFunction() failed\n");
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
    printf("cuModuleGetFunction() failed\n");
    exit(1);
  }
  
  //タプルを初期化する
  init();

  /*全体の実行時間計測*/
  gettimeofday(&begin, NULL);

  /************** block_x * block_y is decided by BLOCK_SIZE. **************/

  /*注意！
   *LEFTをx軸、RIGHTをy軸にした。ほかの場所と逆になっているので余裕があったら修正する
   *
   */

  block_x = arg_left < BLOCK_SIZE_X ? arg_left : BLOCK_SIZE_X;
  block_y = arg_right < BLOCK_SIZE_Y ? arg_right : BLOCK_SIZE_Y;

  grid_x = arg_left / block_x;
  if (arg_left % block_x != 0)
    grid_x++;

  grid_y = arg_right / block_y;
  if (arg_right % block_y != 0)
    grid_y++;

  printf("grid_x = %d\tgrid_y = %d\tblock_y = %d\n",grid_x,grid_y,block_y);

  uint gpu_size = grid_x * grid_y * block_y;
  printf("gpu_size = %d\n",gpu_size);
  
  //malloc memory and 0 for count
  //count = (int *)calloc(gpu_size,sizeof(int));

  /********************************************************************************/

  /********************************************************************
   *a,b,cのメモリを割り当てる。
   *
   */

    /* lt */
  res = cuMemAlloc(&lt_dev, arg_left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lefttuple) failed\n");
    exit(1);
  }
  /* rt */
  res = cuMemAlloc(&rt_dev, arg_right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (righttuple) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&count_dev, gpu_size * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }

  /**********************************************************************************/


  
  /********************** upload lt , rt and count***********************/


  gettimeofday(&time_send_s, NULL);
  res = cuMemcpyHtoD(lt_dev, Tleft, arg_left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (lt) failed: res = %lu\n", res);//conv(res));
    exit(1);
  }
  res = cuMemcpyHtoD(rt_dev, Tright, arg_right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  gettimeofday(&time_send_f, NULL);


  /***************************************************************************/



  /*
    条件に合致するタプルの数をあらかじめ求めておく
    これによってダウンロードするタプルの数を減らせる
   */



  /******************************************************************
    count the number of match tuple
    
  *******************************************************************/

  /*countの時間計測*/
  gettimeofday(&time_count_s, NULL);

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
    printf("cuLaunchKernel(count) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }      

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  

  /***************************************************************************************/



  /**************************** prefix sum *************************************/

  gettimeofday(&time_scan_s, NULL);

  if(!(presum(&count_dev,(uint)gpu_size))){
    printf("count scan error.\n");
    exit(1);
  }

  gettimeofday(&time_scan_f, NULL);

  /********************************************************************/

  if(!transport(count_dev,(uint)gpu_size,&jt_size)){
    printf("transport error.\n");
    exit(1);
  }

  gettimeofday(&time_count_f, NULL);

  /**********************************************************************/



  /************************************************************************
   jt memory alloc and jt upload

  ************************************************************************/

  if(jt_size<=0){
    printf("no tuple is matched.\n");
    exit(1);
  }
  
  res = cuMemAlloc(&jt_dev, jt_size * sizeof(JOIN_TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (join) failed\n");
    exit(1);
  }
  Tjoin = (JOIN_TUPLE *)malloc(jt_size*sizeof(JOIN_TUPLE));

  //実際のjoinの計算時間
  gettimeofday(&time_join_s, NULL);

  void *kernel_args[]={
    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&jt_dev,
    (void *)&count_dev,
    (void *)&arg_left,
    (void *)&arg_right,    
    };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
  res = cuLaunchKernel(
                       function,      // CUfunction f
                       grid_x,        // gridDimX
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


  res = cuMemcpyDtoH(Tjoin, jt_dev, jt_size * sizeof(JOIN_TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (jt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  //downloadの時間計測
  gettimeofday(&time_download_f, NULL);
  
  gettimeofday(&end, NULL);
  /***************************************************************/




  //free GPU memory***********************************************


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

  res = cuMemFree(count_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  /********************************************************************/



  /***********   タプルを送ってダウンロードするまでの時間を計測する*************************/

  printf("\n\n");
  printf("******************execution time********************************************\n\n");

  printf("Calculation all with Devise\n");
  printDiff(begin,end);

  printf("Calculation send with Devise\n");
  printDiff(time_send_s,time_send_f);

  printf("Calculation count with Devise\n");
  printDiff(time_count_s,time_count_f);

  printf("scan time\n");
  printDiff(time_scan_s,time_scan_f);

  printf("Calculation join with Devise\n");
  printDiff(time_join_s,time_join_f);

  printf("Calculation download with Devise\n");
  printDiff(time_download_s,time_download_f);

  /****************************************************************************************/




  //結果を表示したい場合はここ
  //***************************************************************************************

  printf("%d\n",jt_size);


  /*
  for(i=0;i<arg_right;i++){
    printf("%d: %8d%8d\n",i,Tright[i].id,Tright[i].val);

  }

  for(i=0;i<arg_left;i++){
    printf("%d: %8d%8d\n",i,Tleft[i].id,Tleft[i].val);

    }*/


  //ループが何番目かとleftテーブルの最後の値、join後のテーブルのidと最後の値を表示する
  /*
  for(i=0;i<count[grid_x*grid_y*block_y-1];i++){
    if(i%100000==0&&Tjoin[i].lval[0]>0){//if(Tjoin[j * arg_left + i].lval>0){
      printf("%d:left :join: %8d%8d%8d\n",i,Tjoin[i].id, Tjoin[i].lval[VAL_NUM-1], Tjoin[i].rval[VAL_NUM-1]);
    }
  }
  */

  /*****************************************************************************************/
  



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
  free(count);

}


int
main(int argc, char *argv[])
{


  if(argc<4){
    if(argv[1]==NULL){
      printf("argument1 is nothing.\n");
      exit(1);
    }else{
      arg_right=atoi(argv[1]);
      printf("right num :%d\n",arg_right);
    }
    
    if(argv[2]==NULL){
      printf("argument2 is nothing.\n");
      exit(1);
    }else{
      arg_left=atoi(argv[2]);
      printf("left num :%d\n",arg_left);
    }
  }


  //joinの実行
  join();


  return 0;
}
