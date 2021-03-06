#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include <thrust/scan.h>
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


  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て****************************
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
    //gettimeofday(&(Tright[i].t), NULL);
    Tright[i].id = getTupleId();

    for(int j = 0;j<VAL_NUM;j++){
      Tright[i].val[j] = rand()%100; // selectivity = 1.0
    }

  }

  /****************************************************************************/



  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て*******************************
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
    //gettimeofday(&(Tleft[i].t), NULL);    
    Tleft[i].id = getTupleId();

    for(int j = 0; j < VAL_NUM;j++){
      Tleft[i].val[j] = rand()%100; // selectivity = 1.0
    }
    
  }
  
  /*********************************************************************************/


  //JOIN_TUPLEへのGPUでも参照できるメモリの割り当て********************************
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
  int *count,*temp;//maybe long long int is better
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function,s_function;
  CUmodule module,c_module,s_module;
  CUdeviceptr lt_dev, rt_dev, p_dev,count_dev, sum_dev;
  CUdeviceptr ltn_dev, rtn_dev;
  unsigned int block_x, block_y, grid_x, grid_y;
  char fname[256];
  const char *path=".";
  struct timeval tv_cal_s, tv_cal_f,time_join_s,time_join_f,time_upload_s,time_upload_f,time_download_s,time_download_f;
  struct timeval time_count_s,time_count_f,time_Cupload_s,time_Cupload_f,time_Cdownload_s,time_Cdownload_f;
  struct timeval time_scan_s,time_scan_f,time_alloc_s,time_alloc_f,time_free_s,time_free_f;
  struct timeval time_Pupload_s,time_Pupload_f;
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

  
  sprintf(fname, "%s/join_gpuArg.cubin", path);
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
    
  sprintf(fname, "%s/count.cubin", path);
  res = cuModuleLoad(&c_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    exit(1);
  }
  
  res = cuModuleGetFunction(&c_function, c_module, "count");
  //  res = cuModuleGetFunction(&function, module, "_Z3addPjS_S_j");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction() failed\n");
    exit(1);
  }

  sprintf(fname, "%s/presum.cubin", path);
  res = cuModuleLoad(&s_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&s_function, s_module, "presum");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction() failed\n");
    exit(1);
  }  



  
  //タプルを初期化する
  init();


  /*全体の実行時間計測*/
  gettimeofday(&tv_cal_s, NULL);

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

  count = (int *)calloc(grid_x * block_y * grid_y,sizeof(int));


  /********************************************************************************/




  /********************************************************************
   *a,b,cのメモリを割り当てる。
   *
   */

  
  gettimeofday(&time_alloc_s, NULL);
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

  
  res = cuMemAlloc(&count_dev, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&sum_dev, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }

  gettimeofday(&time_alloc_f, NULL);
  /**********************************************************************************/


  
  /********************** upload lt , rt and count***********************/

  /*count uploadの時間計測*/
  gettimeofday(&time_count_s, NULL);
  //gettimeofday(&time_Cupload_s, NULL);

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

  res = cuMemcpyHtoD(count_dev, count, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  //count uploadの時間計測
  //gettimeofday(&time_Cupload_f, NULL);


  /***************************************************************************/



  /*
    条件に合致するタプルの数をあらかじめ求めておく
    これによってダウンロードするタプルの数を減らせる
   */



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
    printf("cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }      

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  } 



  /*
  res = cuMemcpyDtoH(count, count_dev, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  for(int i = 0;i<grid_x*grid_y*block_y;i++){
    printf("%d\n",count[i]);

  }
   
  */

  //count.cuの時間計測
  //gettimeofday(&time_count_f, NULL);


  //count downloadの時間計測
  //gettimeofday(&time_Cdownload_s, NULL);


  /*
  res = cuMemcpyDtoH(&(count[right - 1]), &(count_dev[right - 1]), sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */

  //count downloadの時間計測
  //gettimeofday(&time_Cdownload_f, NULL);    

  /***************************************************************************************/



  /**************************** prefix sum *************************************/
  //thrust::inclusive_scan(count,count+grid_x*grid_y*block_y,count);


  int n = grid_x * grid_y * block_y;
  temp = (int *)calloc(grid_x * block_y * grid_y,sizeof(int));
  int prefix_x = n < SCAN_SIZE_X ? n : SCAN_SIZE_X;
  int scan_grid = n < SCAN_SIZE_X ? 1 : n/SCAN_SIZE_X;
  if (n % prefix_x != 0)
    scan_grid++;


  res = cuMemcpyHtoD(sum_dev, temp, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  void *presum_args[]={    
    (void *)&sum_dev,
    (void *)&count_dev,
    (void *)&n      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  //printf("ok\n");
  res = cuLaunchKernel(
                       s_function,    // CUfunction f
                       scan_grid,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       prefix_x,       // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       presum_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS) {
    printf("cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }      

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(sum) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }  

  /********************************************************************/

  /*
  res = cuMemcpyDtoH(count, sum_dev, grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */

  
  res = cuMemcpyDtoH(&(count[grid_x * grid_y * block_y - 1]), sum_dev + sizeof(int) * (grid_x * grid_y * block_y - 1), sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH (sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  printf("%d\n",count[grid_x * grid_y * block_y -1]);

  //cpy count to GPU again      ***************************************
  /*
  res = cuMemcpyHtoD($(count_dev[grid_x * grid_y * block_y - 1]), $(count[grid_x * grid_y * block_y - 1]), grid_x * grid_y * block_y * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  */

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
    (void *)&sum_dev,
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
    printf("cuCtxSynchronize(join) failed: res = %lu\n", (unsigned long int)res);
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



  /***********   タプルを送ってダウンロードするまでの時間を計測する*************************/
  gettimeofday(&tv_cal_f, NULL);
  
  
  time_cal = (tv_cal_f.tv_sec - tv_cal_s.tv_sec)*1000*1000 + (tv_cal_f.tv_usec - tv_cal_s.tv_usec);
  printf("Calculation all with Devise    : %6f(micro sec)\n", time_cal);
  /*
  time_cal = (time_Cupload_f.tv_sec - time_Cupload_s.tv_sec)*1000*1000 + (time_Cupload_f.tv_usec - time_Cupload_s.tv_usec);
  printf("Calculation Cupload with Devise    : %6f(micro sec)\n", time_cal);

  time_cal = (time_count_f.tv_sec - time_count_s.tv_sec)*1000*1000 + (time_count_f.tv_usec - time_count_s.tv_usec);
  printf("Calculation count with Devise    : %6f(micro sec)\n", time_cal);

  time_cal = (time_Cdownload_f.tv_sec - time_Cdownload_s.tv_sec)*1000*1000 + (time_Cdownload_f.tv_usec - time_Cdownload_s.tv_usec);
  printf("Calculation Cdownload with Devise    : %6f(micro sec)\n", time_cal);
  */
  time_cal = (time_alloc_f.tv_sec - time_alloc_s.tv_sec)*1000*1000 + (time_alloc_f.tv_usec - time_alloc_s.tv_usec);
  printf("Calculation alloc with Devise    : %6f(micro sec)\n", time_cal);

  time_cal = (time_free_f.tv_sec - time_free_s.tv_sec)*1000*1000 + (time_free_f.tv_usec - time_free_s.tv_usec);
  printf("Calculation free with Devise    : %6f(micro sec)\n", time_cal);


  time_cal = (time_count_f.tv_sec - time_count_s.tv_sec)*1000*1000 + (time_count_f.tv_usec - time_count_s.tv_usec);
  printf("Calculation count with Devise    : %6f(micro sec)\n", time_cal);

  time_cal = (time_Pupload_f.tv_sec - time_Pupload_s.tv_sec)*1000*1000 + (time_Pupload_f.tv_usec - time_Pupload_s.tv_usec);
  printf("Calculation Pupload with Devise    : %6f(micro sec)\n", time_cal);

  time_cal = (time_join_f.tv_sec - time_join_s.tv_sec)*1000*1000 + (time_join_f.tv_usec - time_join_s.tv_usec);
  printf("Calculation join with Devise    : %6f(micro sec)\n", time_cal);

  time_cal = (time_download_f.tv_sec - time_download_s.tv_sec)*1000*1000 + (time_download_f.tv_usec - time_download_s.tv_usec);
  printf("Calculation download with Devise    : %6f(micro sec)\n", time_cal);

  /****************************************************************************************/




  //結果を表示したい場合はここ
  //***************************************************************************************
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
