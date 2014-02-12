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
//int *right_tuple_num;
//int *left_tuple_num;
int arg_right;
int arg_left;
#endif


static int
getTupleId(void)
{
  static int id;
  
  return ++id;
}

/*
void getTuple(TUPLE *Tuple){


  CUresult res;

  //メモリ割り当てを行う
  //タプルに初期値を代入

  res = cuMemHostAlloc((void**)&Tuple,*right_tuple_num * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to RIGHT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  for (int i = 0; i < *right_tuple_num; i++) {


    if(&(Tuple[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(Tuple[i]),0,sizeof(TUPLE));

    gettimeofday(&(Tuple[i].t), NULL);

    Tuple[i].id = getTupleId();
    //Tright[i].val = rand() % 100;

    for(int j=0;j<VAL_NUM;j++){
      Tuple[i].val[j] = 1; // selectivity = 1.0
    }


  }


}

*/







//初期化する
void
init(void)
{
  
  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て
  CUresult res;

#ifdef ARG

  //main引数をとる場合、こいつらもGPUのメモリとして確保しておく

  /*  res = cuMemHostAlloc((void**)&right_tuple_num,sizeof(int),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to right_tuple_num failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemHostAlloc((void**)&left_tuple_num,sizeof(int),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to left_tuple_num failed: res = %lu\n", (unsigned long)res);
    exit(1);
    }*/


  /*  right_tuple_num=(int *)malloc(sizeof(int));
  left_tuple_num=(int *)malloc(sizeof(int));
  *right_tuple_num=arg_right;
  *left_tuple_num=arg_left;
  */
  

  //メモリ割り当てを行う
  //タプルに初期値を代入


  /*タプルの初期化を日等の処理にまとめようとしている。うまくいかーん
  getTuple(Tright);
  getTuple(Tleft);
  */

  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て
  res = cuMemHostAlloc((void**)&Tright,arg_right * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to RIGHT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  for (int i = 0; i < arg_right; i++) {


    if(&(Tright[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(Tright[i]),0,sizeof(TUPLE));

    gettimeofday(&(Tright[i].t), NULL);

    Tright[i].id = getTupleId();
    //Tright[i].val = rand() % 100;

    for(int j=0;j<VAL_NUM;j++){
      Tright[i].val[j] = 1; // selectivity = 1.0
    }
    //Tright[i].val[0]=1;

    }

  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て
  res = cuMemHostAlloc((void**)&Tleft,arg_left * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  for (int i = 0; i < arg_left; i++) {
    
    
    if(&(Tleft[i])==NULL){
      printf("left TUPLE allocate error.\n");
      exit(1);
    }
    
    //０に初期化
    memset(&(Tleft[i]),0,sizeof(TUPLE));
    
    
    gettimeofday(&(Tleft[i].t), NULL);
    
    
    Tleft[i].id = getTupleId();
    //Tleft[i].val = rand() % 100;
    for(int j=0;j<VAL_NUM;j++){
      Tleft[i].val[j] = 1; // selectivity = 1.0
    }
    
  }
  

  //JOIN_TUPLEへのGPUでも参照できるメモリの割り当て
  res = cuMemHostAlloc((void**)&Tjoin,arg_right * arg_left * sizeof(JOIN_TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to JOIN_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  for(int i=0;i < arg_left * arg_right;i++){


    if(&(Tjoin[i])==NULL){
      printf("left TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(Tjoin[i]),0,sizeof(JOIN_TUPLE));


    Tjoin[i].id = getTupleId();
  }

#else  // ifdef ARG
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

    for(int k=0;k<VAL_NUM;k++){
      Tright[i].val[j] = 1; // selectivity = 1.0
    }


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
    for(int k=0;k<VAL_NUM;k++){
      Tleft[i].val[j] = 1; // selectivity = 1.0
    }

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

#endif




}


//メモリ解放のため新しく追加した関数。バグがあるかも
void
tuple_free(void){


  int i;

#ifdef ARG

  //cuMemFreeHost(left_tuple_num);
  //cuMemFreeHost(right_tuple_num);
  cuMemFreeHost(Tright);
  cuMemFreeHost(Tleft);
  cuMemFreeHost(Tjoin);

  /*
  for(i=0;i<*right_tuple_num;i++){
    cuMemFreeHost(Tright);

  }

  for(i=0;i<*left_tuple_num;i++){
    cuMemFreeHost(Tleft);

  }

  for(i=0;i<*right_tuple_num * *left_tuple_num;i++){
    cuMemFreeHost(Tjoin);

    }*/

#else

  cuMemFreeHost(Tright);
  cuMemFreeHost(Tleft);
  cuMemFreeHost(Tjoin);


  /*
  for(i=0;i<MAX_RIGHT;i++){
    cuMemFreeHost(Tright);

  }

  for(i=0;i<MAX_LEFT;i++){
    cuMemFreeHost(Tleft);

  }

  for(i=0;i<MAX_RIGHT*MAX_LEFT;i++){
    cuMemFreeHost(Tjoin);

    }*/

#endif
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
  unsigned int block_x, block_y, grid_x, grid_y;
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

  // res = cuFuncSetSharedSize(function, 0x40); /* just random */
  // if (res != CUDA_SUCCESS) {
  //   printf("cuFuncSetSharedSize() failed\n");
  //   exit(1);
  // }



  //タプルを初期化する
  init();

  gettimeofday(&tv_cal_s, NULL);


  /* block_x * block_y is decided by BLOCK_SIZE. */

  /*注意！
   *LEFTをx軸、RIGHTをy軸にした。ほかの場所と逆になっているので余裕があったら修正する
   *
   */

#ifdef ARG

  block_x = arg_left < BLOCK_SIZE_X ? arg_left : BLOCK_SIZE_X;
  block_y = arg_right < BLOCK_SIZE_Y ? arg_right : BLOCK_SIZE_Y;

  grid_x = arg_left / block_x;
  if (arg_left % block_x != 0)
    grid_x++;

  grid_y = arg_right / block_y;
  if (arg_right % block_y != 0)
    grid_y++;


#else

  block_x = MAX_LEFT < BLOCK_SIZE_X ? MAX_LEFT : BLOCK_SIZE_X;
  block_y = MAX_RIGHT < BLOCK_SIZE_Y ? MAX_RIGHT : BLOCK_SIZE_Y;
  grid_x = MAX_LEFT / block_x;
  if (MAX_LEFT % block_x != 0)
    grid_x++;
  grid_y = MAX_RIGHT / block_y;
  if (MAX_RIGHT % block_y != 0)
    grid_y++;

#endif


  cuMemHostGetDevicePointer(&lt_dev,Tleft,0);
  cuMemHostGetDevicePointer(&rt_dev,Tright,0);
  cuMemHostGetDevicePointer(&p_dev,Tjoin,0);


  //main引数をとる場合、それを送る
#ifdef ARG


  // cuMemHostGetDevicePointer(&ltn_dev,left_tuple_num,0); 
  // cuMemHostGetDevicePointer(&rtn_dev,right_tuple_num,0);
  
  void *kernel_args[]={

    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&p_dev,
    // (void *)&ltn_dev,
    // (void *)&rtn_dev
    (void *)&arg_left,
    (void *)&arg_right,

    
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
    printf("cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  
  
  gettimeofday(&tv_cal_f, NULL);
    
  time_cal = (tv_cal_f.tv_sec - tv_cal_s.tv_sec)*1000*1000 + (tv_cal_f.tv_usec - tv_cal_s.tv_usec);
  printf("Calculation with Devise    : %6f(micro sec)\n", time_cal);

#else

  void *kernel_args[]={

    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&p_dev

  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  cuLaunchKernel(function,grid_x,grid_y,1,block_x,block_y,1,0,NULL,kernel_args,NULL);

  cuCtxSynchronize();
  
  gettimeofday(&tv_cal_f, NULL);

  time_cal = (tv_cal_f.tv_sec - tv_cal_s.tv_sec)*1000*1000 + (tv_cal_f.tv_usec - tv_cal_s.tv_usec);
  printf("Calculation with Devise    : %6f(micro sec)\n", time_cal);

#endif



#ifdef ARG

  /*for(i=0;i<*right_tuple_num;i++){
    printf("%d: %8d%8d\n",i,Tright[i].id,Tright[i].val);

  }

  for(i=0;i<*left_tuple_num;i++){
    printf("%d: %8d%8d\n",i,Tleft[i].id,Tleft[i].val);

    }*/
  /*
  for(i=0;i<*left_tuple_num;i++){
    for(j=0;j<*right_tuple_num;j++){
      if((j * *left_tuple_num + i)%PER_SHOW==0){//if(Tjoin[j * *left_tuple_num + i].lval>0){
        printf("%d:left %8d :join: %8d%8d%8d\n",j * *left_tuple_num + i,Tleft[i].val[VAL_NUM-1],Tjoin[j * *left_tuple_num + i].id, Tjoin[j * *left_tuple_num + i].lval[VAL_NUM-1], Tjoin[j * *left_tuple_num + i].rval[VAL_NUM-1]);
      }
    }
    }*/
#else

  /*
  for(i=0;i<MAX_RIGHT;i++){
    printf("%d: %8d%8d\n",i,Tright[i].id,Tright[i].val);

  }

  for(i=0;i<MAX_LEFT;i++){
    printf("%d: %8d%8d\n",i,Tleft[i].id,Tleft[i].val);

  }

  for(i=0;i<MAX_RIGHT*MAX_LEFT;i++){
    printf("%d: %8d%8d%8d\n",i,Tjoin[i].id, Tjoin[i].lval, Tjoin[i].rval);

    }*/

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
