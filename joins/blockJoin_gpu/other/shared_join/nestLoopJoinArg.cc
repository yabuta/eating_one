#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include "tuple.h"

TUPLE *Tright;
TUPLE *Tleft;
JOIN_TUPLE *Tjoin;

//main引数を受け取るグローバル変数
int arg_right;
int arg_left;

extern char *conv(unsigned int res);

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


  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て
  res = cuMemHostAlloc((void**)&Tright,arg_right * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to RIGHT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  srand((unsigned)time(NULL));      

  for (int i = 0; i < arg_right; i++) {


    if(&(Tright[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }

    //０に初期化
    memset(&(Tright[i]),0,sizeof(TUPLE));

    gettimeofday(&(Tright[i].t), NULL);

    Tright[i].id = getTupleId();

    for(int j = 0;j<VAL_NUM;j++){
      Tright[i].val[j] = rand()%100; // selectivity = 1.0
    }

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
    for(int j = 0; j < VAL_NUM;j++){
      Tleft[i].val[j] = rand()%100; // selectivity = 1.0
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
  int *count,*pre;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function;
  CUmodule module,c_module;
  CUdeviceptr lt_dev, rt_dev, p_dev,count_dev, pre_dev;
  CUdeviceptr ltn_dev, rtn_dev;
  unsigned int block_x, block_y, grid_x, grid_y;
  char fname[256];
  const char *path=".";
  struct timeval tv_cal_s, tv_cal_f,time_join_s,time_join_f,time_upload_s,time_upload_f,time_download_s,time_download_f;
  double time_cal;

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
   *今回はjoin_gpu.cubinとcountJoinTuple.cubinの二つの関数を実行する
   */

  
  sprintf(fname, "%s/join_gpuArg.cubin", path);
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

  /*  
  sprintf(fname, "%s/count.cubin", path);
  res = cuModuleLoad(&c_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    exit(1);
  }

  */
  /*
  res = cuModuleGetFunction(&c_function, c_module, "count");
  //  res = cuModuleGetFunction(&function, module, "_Z3addPjS_S_j");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction() failed\n");
    exit(1);
  }
  */
  /*
  sprintf(fname, "%s/prefix.cubin", path);
  res = cuModuleLoad(&pre_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&pre_fun, pre_module, "prefix");
  //  res = cuModuleGetFunction(&function, module, "_Z3addPjS_S_j");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction() failed\n");
    exit(1);
  }
  */


  //タプルを初期化する
  init();

  //printf("ok\n");

  /*全体の実行時間計測*/
  gettimeofday(&tv_cal_s, NULL);

  /* block_x * block_y is decided by BLOCK_SIZE. */

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

  unsigned int grid = grid_x * grid_y;

  
  count = (int *)calloc(arg_left * arg_right,sizeof(int));
  /*for(int i=0;i<arg_left*arg_right;i++){
    count[i]=0;
    }*/
  
  pre = (int *)calloc(arg_left,sizeof(int));
  /*for(int i = 0; i<arg_left ;i++){
    pre[i] = 0;
    }*/

  /*
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

  /* p */
  
  res = cuMemAlloc(&p_dev, arg_left * arg_right * sizeof(JOIN_TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (jointuple) failed\n");
    exit(1);
  }
  

  //printf("ok\n");
  /*
  res = cuMemAlloc(&count_dev, arg_left * arg_right * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }
  
  res = cuMemAlloc(&pre_dev, arg_left * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (pre) failed\n");
    exit(1);
  }
  */

  //printf("ok\n");

  /* upload lt , rt and p*/

  /*uploadの時間計測*/
  gettimeofday(&time_upload_s, NULL);

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

  res = cuMemcpyHtoD(p_dev, Tjoin, arg_left * arg_right * sizeof(JOIN_TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (join) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  //uploadの時間計測
  gettimeofday(&time_upload_f, NULL);

  /*  
  res = cuMemcpyHtoD(count_dev, count, arg_left * arg_right * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemcpyHtoD(pre_dev, pre, arg_left * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (pre) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */

  /*
    条件に合致するタプルの数をあらかじめ求めておく
    これによってダウンロードするタプルの数を減らせる
   */

  /*
  void *kernel_args[]={
    
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


  void *kernel_args[]={
    
    (void *)&count_dev,
    (void *)&pre_dev,
    (void *)&arg_left,
    (void *)&arg_right
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  //printf("ok\n");
  res = cuLaunchKernel(
                       pre_fun,    // CUfunction f
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


  res = cuMemcpyDtoH(count, count_dev, arg_right * arg_left * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (p) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  */

  /*
  for(int i=0; i<arg_left * arg_right ;i++){
    printf("%d\n",count[i]);
    }*/
  


  //引数をとる場合、それを送る


  
  //実際のjoinの計算時間
  gettimeofday(&time_join_s, NULL);

  void *kernel_args[]={

    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&p_dev,
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


  res = cuMemcpyDtoH(Tjoin, p_dev, arg_left *arg_right * sizeof(JOIN_TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (p) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  //downloadの時間計測
  gettimeofday(&time_download_f, NULL);
  

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

  /*
  res = cuMemFree(count_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(pre_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (pre_dev) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */

  /*
  for(int i = 0;i < grid;i++){
    printf("%d\n",count[i]);
    }*/

  /*タプルを送ってダウンロードするまでの時間を計測する*/
  gettimeofday(&tv_cal_f, NULL);
    
  time_cal = (tv_cal_f.tv_sec - tv_cal_s.tv_sec)*1000*1000 + (tv_cal_f.tv_usec - tv_cal_s.tv_usec);
  printf("Calculation all with Devise    : %6f(micro sec)\n", time_cal);
  time_cal = (time_upload_f.tv_sec - time_upload_s.tv_sec)*1000*1000 + (time_upload_f.tv_usec - time_upload_s.tv_usec);
  printf("Calculation upload with Devise    : %6f(micro sec)\n", time_cal);
  time_cal = (time_join_f.tv_sec - time_join_s.tv_sec)*1000*1000 + (time_join_f.tv_usec - time_join_s.tv_usec);
  printf("Calculation join with Devise    : %6f(micro sec)\n", time_cal);
  time_cal = (time_download_f.tv_sec - time_download_s.tv_sec)*1000*1000 + (time_download_f.tv_usec - time_download_s.tv_usec);
  printf("Calculation download with Devise    : %6f(micro sec)\n", time_cal);



  //結果を表示したい場合はここ
  //
  /*
  for(i=0;i<arg_right;i++){
    printf("%d: %8d%8d\n",i,Tright[i].id,Tright[i].val);

  }

  for(i=0;i<arg_left;i++){
    printf("%d: %8d%8d\n",i,Tleft[i].id,Tleft[i].val);

    }*/

  //ループが何番目かとleftテーブルの最後の値、join後のテーブルのidと最後の値を表示する
      
  for(i=0;i<arg_left;i++){
    for(j=0;j<arg_right;j++){
      if((j * arg_left + i)%10000000==0){//if(Tjoin[j * arg_left + i].lval>0){
        printf("%d:left %8d :join: %8d%8d%8d\n",j * arg_left + i,Tleft[i].val[VAL_NUM-1],Tjoin[j * arg_left + i].id, Tjoin[j * arg_left + i].lval[VAL_NUM-1], Tjoin[j * arg_left + i].rval[VAL_NUM-1]);
      }
    }
  }


  
  res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload module failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  
  /*
  res = cuModuleUnload(c_module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload c_module failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  */

  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  //割り当てたメモリを開放する
  tuple_free();
  //free(count);
  //free(pre);

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
