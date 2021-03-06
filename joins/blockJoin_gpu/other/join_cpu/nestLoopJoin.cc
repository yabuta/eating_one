#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include "tuple.h"

TUPLE *Tright;
TUPLE *Tleft;
JOIN_TUPLE *Tjoin;

static int
getTupleId(void)
{
  static int id;
  
  return ++id;
}

//構造体を初期化する。MAX_LRFTまで作る？構造体はリストになってる模様
void
init(void)
{
  
  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て
  CUresult res;

  res = cuMemHostAlloc((void**)&Tright,MAX_RIGHT * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to RIGHT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  for (int i = 0; i < MAX_RIGHT; i++) {


    if(&(Tright[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(Tright[i]),0,sizeof(TUPLE));

    //ここまで・・・・・・・・・・・・・・・


    gettimeofday(&(Tright[i].t), NULL);

    Tright[i].id = getTupleId();
    //p->val = rand() % 100;
    Tright[i].val = 1; // selectivity = 1.0

  }


  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て
  res = cuMemHostAlloc((void**)&Tleft,MAX_LEFT * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  for (int i = 0; i < MAX_LEFT; i++) {


    if(&(Tleft[i])==NULL){
      printf("left TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(Tleft[i]),0,sizeof(TUPLE));

    //ここまで・・・・・・・・・・・・・・・

    gettimeofday(&(Tleft[i].t), NULL);


    Tleft[i].id = getTupleId();
    //p->val = rand() % 100;
    Tleft[i].val = 1; // selectivity = 1.0
  }
  

  //JOIN_TUPLEへのGPUでも参照できるメモリの割り当て
  res = cuMemHostAlloc((void**)&Tjoin,MAX_RIGHT * MAX_LEFT * sizeof(JOIN_TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to JOIN_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  for(int i=0;i < MAX_LEFT*MAX_RIGHT;i++){


    if(&(Tjoin[i])==NULL){
      printf("left TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(Tjoin[i]),0,sizeof(JOIN_TUPLE));

    //ここまで・・・・・・・・・・・・・・・

    Tjoin[i].id = getTupleId();
  }




}


//メモリ解放のため新しく追加した関数。バグがあるかも
void
tuple_free(void){


  int i;


  for(i=0;i<MAX_RIGHT;i++){
    cuMemFreeHost(Tright);

  }

  for(i=0;i<MAX_LEFT;i++){
    cuMemFreeHost(Tleft);

  }

  for(i=0;i<MAX_RIGHT*MAX_LEFT;i++){
    cuMemFreeHost(Tjoin);

  }

}



//HrightとHleftをそれぞれ比較する。GPUで並列化するforループもここにあるもので行う。
void
join()
{

  int i, j, idx;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function;
  CUmodule module;
  CUdeviceptr lt_dev, rt_dev, p_dev;
  int block_x, block_y, grid_x, grid_y;
  char fname[256];

  const char *path=".";

  struct timeval tv_HtoD_s, tv_HtoD_f, tv_cal_s, tv_cal_f, tv_DtoH_s, tv_DtoH_f;
  double time_HtoD, time_cal, time_DtoH;

  /* block_x * block_y should not exceed 512. */

  /*注意！
   *LEFTをx軸、RIGHTをy軸にした。ほかの場所と逆になっているので余裕があったら修正する
   *
   */

  block_x = MAX_LEFT < 16 ? MAX_LEFT : 16;
  block_y = MAX_RIGHT < 16 ? MAX_RIGHT : 16;
  grid_x = MAX_LEFT / block_x;
  if (MAX_LEFT % block_x != 0)
    grid_x++;
  grid_y = MAX_RIGHT / block_y;
  if (MAX_RIGHT % block_y != 0)
    grid_y++;



  /*この辺で初期化*/
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



  //タプルを初期化する
  init();

  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *
   */
  sprintf(fname, "%s/join_gpu.cubin", path);
  res = cuModuleLoad(&module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&function, module, "join");
  //  res = cuModuleGetFunction(&function, module, "_Z3addPjS_S_j");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction() failed\n");
    exit(1);
  }
  res = cuFuncSetSharedSize(function, 0x40); /* just random */
  if (res != CUDA_SUCCESS) {
    printf("cuFuncSetSharedSize() failed\n");
    exit(1);
  }

  cuMemHostGetDevicePointer(&lt_dev,Tleft,0);
  cuMemHostGetDevicePointer(&rt_dev,Tright,0);
  cuMemHostGetDevicePointer(&p_dev,Tjoin,0);


  void *kernel_args[]={

    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&p_dev

  };


  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
  gettimeofday(&tv_cal_s, NULL);

  cuLaunchKernel(function,grid_x,grid_y,1,block_x,block_y,1,0,NULL,kernel_args,NULL);

  cuCtxSynchronize();
  
  gettimeofday(&tv_cal_f, NULL);

  time_cal = (tv_cal_f.tv_sec - tv_cal_s.tv_sec)*1000*1000 + (tv_cal_f.tv_usec - tv_cal_s.tv_usec);
  printf("Calculation with Devise    : %6f(micro sec)\n", time_cal);

  res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  
  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  //割り当てたメモリを開放する
  tuple_free();


}


int
main(int argc, char *argv[])
{

  //joinの実行
  join();

  return 0;
}
