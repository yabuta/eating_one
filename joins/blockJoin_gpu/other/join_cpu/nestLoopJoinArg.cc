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

#ifdef ARG
int *right_tuple_num;
int *left_tuple_num;
int arg_right;
int arg_left;
#endif


static int
getTupleId(void)
{
  static int id;
  
  return ++id;
}

static TUPLE *
getTuple(void)
{
  TUPLE *p;
  
  if (!(p = (TUPLE *)calloc(1, sizeof(TUPLE)))) ERR;
  gettimeofday(&p->t, NULL);
  p->id = getTupleId();
  //p->val = rand() % 100;
  p->val = 1; // selectivity = 1.0

  return p;
}

static JOIN_TUPLE *
getJoinTuple(TUPLE *lt, TUPLE *rt)
{
  JOIN_TUPLE *p;

  if (!(p = (JOIN_TUPLE *)calloc(1, sizeof(JOIN_TUPLE)))) ERR;
  //gettimeofday(&p->t, NULL);
  p->id = getTupleId();
  p->lval = lt->val;
  p->rval = rt->val;  
  // lid & rid are just for debug
  p->lid = lt->id;
  p->rid = rt->id;

  //printf("New Tuple %8d%8d%8d%8d\n", p->rid, p->lid, p->rval, p->lval);
  
  return p;
}


//構造体を初期化する。MAX_LRFTまで作る？構造体はリストになってる模様


void
init(void)
{
  bzero(&Hright, sizeof(TUPLE));
  bzero(&Hleft, sizeof(TUPLE));  
  bzero(&Hjoin, sizeof(JOIN_TUPLE));

  Tright = &Hright;
  Tleft = &Hleft;
  Tjoin = &Hjoin;

  for (int i = 0; i < MAX_LEFT; i++) {
    Tright->nxt = getTuple();
    Tright = Tright->nxt;
  }

  for (int i = 0; i < MAX_LEFT; i++) {
    Tleft->nxt = getTuple();
    Tleft = Tleft->nxt;
  }
}

//メモリ解放のため新しく追加した関数。バグがあるかも
void
tuple_free(void){

  Tright = &Hright;
  Tleft = &Hleft;
  Tjoin = &Hjoin;

  TUPLE *temp;
  JOIN_TUPLE *jtemp;

  if(Tright->nxt!=NULL) Tright=Tright->nxt;
  if(Tleft->nxt!=NULL) Tleft=Tleft->nxt;
  if(Tjoin->nxt!=NULL) Tjoin=Tjoin->nxt;

  while(Tright->nxt!=NULL){

    temp=Tright->nxt;
    free(Tright);
    Tright=temp;

  }
  free(Tright);


  while(Tleft->nxt!=NULL){

    temp=Tleft->nxt;
    free(Tleft);
    Tleft=temp;

  }
  free(Tleft);

  while(Tjoin->nxt!=NULL){

    jtemp=Tjoin->nxt;
    free(Tjoin);
    Tjoin=jtemp;

  }
  free(Tjoin);

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
#ifdef ARG
  CUdeviceptr ltn_dev, rtn_dev;
#endif
  int block_x, block_y, grid_x, grid_y;
  char fname[256];

  const char *path=".";

  struct timeval tv_HtoD_s, tv_HtoD_f, tv_cal_s, tv_cal_f, tv_DtoH_s, tv_DtoH_f;
  double time_HtoD, time_cal, time_DtoH;



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



  /* block_x * block_y should not exceed 512. */

  /*注意！
   *LEFTをx軸、RIGHTをy軸にした。ほかの場所と逆になっているので余裕があったら修正する
   *
   */

#ifdef ARG

  block_x = *left_tuple_num < 16 ? *left_tuple_num : 16;
  block_y = *right_tuple_num < 16 ? *right_tuple_num : 16;
  grid_x = *left_tuple_num / block_x;
  if (*left_tuple_num % block_x != 0)
    grid_x++;
  grid_y = *right_tuple_num / block_y;
  if (*right_tuple_num % block_y != 0)
    grid_y++;

#else

  block_x = MAX_LEFT < 16 ? MAX_LEFT : 16;
  block_y = MAX_RIGHT < 16 ? MAX_RIGHT : 16;
  grid_x = MAX_LEFT / block_x;
  if (MAX_LEFT % block_x != 0)
    grid_x++;
  grid_y = MAX_RIGHT / block_y;
  if (MAX_RIGHT % block_y != 0)
    grid_y++;

#endif

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


  //main引数をとる場合、それを送る
#ifdef ARG

  cuMemHostGetDevicePointer(&ltn_dev,left_tuple_num,0); 
  cuMemHostGetDevicePointer(&rtn_dev,right_tuple_num,0);
  
  void *kernel_args[]={

    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&p_dev,
    (void *)&ltn_dev,
    (void *)&rtn_dev

  };

#else

  void *kernel_args[]={

    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&p_dev

  };

#endif

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
  gettimeofday(&tv_cal_s, NULL);

  cuLaunchKernel(function,grid_x,grid_y,1,block_x,block_y,1,0,NULL,kernel_args,NULL);

  cuCtxSynchronize();
  
  gettimeofday(&tv_cal_f, NULL);

  time_cal = (tv_cal_f.tv_sec - tv_cal_s.tv_sec)*1000*1000 + (tv_cal_f.tv_usec - tv_cal_s.tv_usec);
  printf("Calculation with Devise    : %6f(micro sec)\n", time_cal);


#ifdef ARG

  /*for(i=0;i<*right_tuple_num;i++){
    printf("%d: %8d%8d\n",i,Tright[i].id,Tright[i].val);

  }

  for(i=0;i<*left_tuple_num;i++){
    printf("%d: %8d%8d\n",i,Tleft[i].id,Tleft[i].val);

    }*/

  for(i=0;i<*left_tuple_num;i++){
    for(j=0;j<*right_tuple_num;j++){
      if(Tjoin[j * *left_tuple_num + i].lval>0){//if(i%PER_SHOW==0){
        printf("%d:left %8d :join: %8d%8d%8d\n",j * *left_tuple_num + i,Tleft[i].val,Tjoin[j * *left_tuple_num + i].id, Tjoin[j * *left_tuple_num + i].lval, Tjoin[j * *left_tuple_num + i].rval);
      }
    }
  }
#else
  for(i=0;i<MAX_RIGHT;i++){
    printf("%d: %8d%8d\n",i,Tright[i].id,Tright[i].val);

  }

  for(i=0;i<MAX_LEFT;i++){
    printf("%d: %8d%8d\n",i,Tleft[i].id,Tleft[i].val);

  }

  for(i=0;i<MAX_RIGHT*MAX_LEFT;i++){
    printf("%d: %8d%8d%8d\n",i,Tjoin[i].id, Tjoin[i].lval, Tjoin[i].rval);

  }

#endif


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

#ifdef ARG


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

#endif

  //joinの実行
  join();


  return 0;
}
