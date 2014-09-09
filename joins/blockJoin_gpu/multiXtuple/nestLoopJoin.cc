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

void shuffle(TUPLE ary[],int size) {    
  srand((unsigned)time(NULL));
  for(int i=0;i<size;i++){
    int j = rand()%size;
    int t = ary[i].val[0];
    ary[i].val[0] = ary[j].val[0];
    ary[j].val[0] = t;
  }
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
  uint *used;//usedなnumberをstoreする
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  for(uint i=0; i<SELECTIVITY ;i++){
    used[i] = i;
  }
  uint selec = SELECTIVITY;

  //uniqueなnumberをvalにassignする
  for (uint i = 0; i < arg_right; i++) {
    if(&(Tright[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }
    Tright[i].id = getTupleId();
    uint temp = rand()%selec;
    uint temp2 = used[temp];
    selec = selec-1;
    used[temp] = used[selec];

    for(uint j= 0 ; j<VAL_NUM ; j++){
      Tright[i].val[j] = temp2; 
    }
  }


  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て*******************************
  res = cuMemHostAlloc((void**)&Tleft,arg_left * sizeof(TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  uint counter = 0;//matchするtupleをcountする。
  uint *used_r;
  used_r = (uint *)calloc(arg_right,sizeof(uint));
  for(uint i=0; i<arg_right ; i++){
    used_r[i] = i;
  }
  uint rg = arg_right;
  uint l_diff;//
  if(MATCH_RATE != 0){
    l_diff = arg_left/(MATCH_RATE*arg_right);
  }else{
    l_diff = 1;
  }
  for (uint i = 0; i < arg_left; i++) {
    Tleft[i].id = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*arg_right){
      uint temp = rand()%rg;
      uint temp2 = used_r[temp];
      rg = rg-1;
      used[temp] = used[rg];

      for(uint j= 0 ; j<VAL_NUM ; j++){
        Tleft[i].val[j] = Tright[temp2].val[j];      
      }
      counter++;
    }else{
      uint temp = rand()%selec;
      uint temp2 = used[temp];
      selec = selec-1;
      used[temp] = used[selec];
      for(uint j= 0 ; j<VAL_NUM ; j++){
        Tleft[i].val[j] = temp2; 
      }
    }
  }

  printf("%d\n",counter);

  
  free(used);
  free(used_r);

  shuffle(Tleft,arg_left);

  res = cuMemHostAlloc((void**)&Tjoin,JT_SIZE * sizeof(JOIN_TUPLE),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
     
}



//メモリ解放のため新しく追加した関数。バグがあるかも
void
tuple_free(void){

  if(Tright != NULL){
    cuMemFreeHost(Tright);
  }
  if(Tleft != NULL){
    cuMemFreeHost(Tleft);
  }
  if(Tjoin != NULL){
    cuMemFreeHost(Tjoin);
  }
}



//HrightとHleftをそれぞれ比較する。GPUで並列化するforループもここにあるもので行う。
void
join()
{

  int i, j, idx;
  //int *count;//maybe long long int is better
  uint jt_size,gpu_size;
  uint total = 0;
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
  long join_time=0,count_time=0,scan_time=0,send_time=0,down_time=0;
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
  res = cuModuleGetFunction(&c_function, module, "count");
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
  /*
  block_x = PART < BLOCK_SIZE_X ? PART : BLOCK_SIZE_X;
  block_y = arg_right < BLOCK_SIZE_Y ? arg_right : BLOCK_SIZE_Y;

  grid_x = arg_left / block_x;
  if (arg_left % block_x != 0)
    grid_x++;

  grid_y = arg_right / block_y;
  if (arg_right % block_y != 0)
    grid_y++;

  block_y = 1;
  */

  block_x = BLOCK_SIZE_X;
  block_y = BLOCK_SIZE_Y;
  grid_x = PART / (X_NUM*block_x);
  if (PART % (X_NUM*block_x) != 0)
    grid_x++;

  grid_y = PART / block_y;
  if (PART % block_y != 0)
    grid_y++;

  block_y = 1;
  printf("grid_x = %d\tgrid_y = %d\tblock_x = %d\tblock_y = %d\n",grid_x,grid_y,block_x,block_y);

  gpu_size = grid_x * grid_y * block_x * block_y;
  printf("gpu_size = %d\n",gpu_size);
  if(gpu_size>MAX_LARGE_ARRAY_SIZE){
    gpu_size = MAX_LARGE_ARRAY_SIZE * iDivUp(gpu_size,MAX_LARGE_ARRAY_SIZE);
  }else if(gpu_size > MAX_SHORT_ARRAY_SIZE){
    gpu_size = MAX_SHORT_ARRAY_SIZE * iDivUp(gpu_size,MAX_SHORT_ARRAY_SIZE);
  }else{
    gpu_size = MAX_SHORT_ARRAY_SIZE;
  }


  /********************************************************************************/

  res = cuMemAlloc(&lt_dev, PART * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lefttuple) failed\n");
    exit(1);
  }
  res = cuMemAlloc(&rt_dev, PART * sizeof(TUPLE));
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

  for(uint ll = 0; ll < arg_left ; ll += PART){
    for(uint rr = 0; rr < arg_right ; rr += PART){

      uint lls=PART,rrs=PART;
      if((ll+PART) >= arg_left){
        lls = arg_left - ll;
      }
      if((rr+PART) >= arg_right){
        rrs = arg_right - rr;
      }

      block_x = lls < X_NUM*BLOCK_SIZE_X ? lls : BLOCK_SIZE_X;
      block_y = rrs < BLOCK_SIZE_Y ? rrs : BLOCK_SIZE_Y;      
      grid_x = lls / (X_NUM*block_x);
      if (lls % (X_NUM*block_x) != 0)
        grid_x++;      
      grid_y = rrs / block_y;
      if (rrs % block_y != 0)
        grid_y++;      
      block_y = 1;

      printf("\nStarting...\nll = %d\trr = %d\tlls = %d\trrs = %d\n",ll,rr,lls,rrs);
      printf("grid_x = %d\tgrid_y = %d\tblock_x = %d\tblock_y = %d\n",grid_x,grid_y,block_x,block_y);
      gpu_size = grid_x * grid_y * block_x * block_y+1;
      printf("gpu_size = %d\n",gpu_size);


      gettimeofday(&time_send_s, NULL);
      res = cuMemcpyHtoD(lt_dev, &(Tleft[ll]), lls * sizeof(TUPLE));
      if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD (lt) failed: res = %lu\n", res);//conv(res));
        exit(1);
      }
      res = cuMemcpyHtoD(rt_dev, &(Tright[rr]), rrs * sizeof(TUPLE));
      if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
        exit(1);
      }
      gettimeofday(&time_send_f, NULL);

      send_time += (time_send_f.tv_sec - time_send_s.tv_sec) * 1000 * 1000 + (time_send_f.tv_usec - time_send_s.tv_usec);

      /******************************************************************
    count the number of match tuple
    
      *******************************************************************/
      
      /*countの時間計測*/
      gettimeofday(&time_count_s, NULL);

      void *count_args[]={
    
        (void *)&lt_dev,
        (void *)&rt_dev,
        (void *)&count_dev,
        (void *)&lls,
        (void *)&rrs
        
      };
      
      //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
      
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
      scan_time += (time_scan_f.tv_sec - time_scan_s.tv_sec) * 1000 * 1000 + (time_scan_f.tv_usec - time_scan_s.tv_usec);      

      /********************************************************************/      
      if(!transport(count_dev,(uint)gpu_size,&jt_size)){
        printf("transport error.\n");
        exit(1);
      }

      gettimeofday(&time_count_f, NULL);

      count_time += (time_count_f.tv_sec - time_count_s.tv_sec) * 1000 * 1000 + (time_count_f.tv_usec - time_count_s.tv_usec);

      /**********************************************************************/



      /************************************************************************
      jt memory alloc and jt upload

      ************************************************************************/

      if(jt_size<=0){
        //printf("no tuple is matched.\n");
        total += jt_size;
        //printf("End...\n jt_size = %d\ttotal = %d\n",jt_size,total);
        jt_size = 0;
      }else{
        res = cuMemAlloc(&jt_dev, jt_size * sizeof(JOIN_TUPLE));
        if (res != CUDA_SUCCESS) {
          printf("cuMemAlloc (join) failed\n");
          exit(1);
        }      
        //実際のjoinの計算時間
        gettimeofday(&time_join_s, NULL);
        
        void *kernel_args[]={
          (void *)&lt_dev,
          (void *)&rt_dev,
          (void *)&jt_dev,
          (void *)&count_dev,
          (void *)&lls,
          (void *)&rrs,    
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
        
      
        /*実際のjoinの計算時間*/
        gettimeofday(&time_join_f, NULL);
        join_time += (time_join_f.tv_sec - time_join_s.tv_sec) * 1000 * 1000 + (time_join_f.tv_usec - time_join_s.tv_usec);

        //downloadの時間計測
        gettimeofday(&time_download_s, NULL);      
        
        res = cuMemcpyDtoH(&(Tjoin[total]), jt_dev, jt_size * sizeof(JOIN_TUPLE));
        if (res != CUDA_SUCCESS) {
          printf("cuMemcpyDtoH (jt) failed: res = %lu\n", (unsigned long)res);
          exit(1);
        }
        
        //downloadの時間計測
        gettimeofday(&time_download_f, NULL);

        down_time += (time_download_f.tv_sec - time_download_s.tv_sec) * 1000 * 1000 + (time_download_f.tv_usec - time_download_s.tv_usec);        

        cuMemFree(jt_dev);
        total += jt_size;
        printf("End...\n jt_size = %d\ttotal = %d\n",jt_size,total);
        jt_size = 0;
        
      }
    }
    

  }

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
  printf("Diff: %ld us (%ld ms)\n", send_time, send_time/1000);

  printf("Calculation count with Devise\n");
  printf("Diff: %ld us (%ld ms)\n", count_time, count_time/1000);

  printf("scan time\n");
  printf("Diff: %ld us (%ld ms)\n", scan_time, scan_time/1000);

  printf("Calculation join with Devise\n");
  printf("Diff: %ld us (%ld ms)\n", join_time, join_time/1000);

  printf("Calculation download with Devise\n");
  printf("Diff: %ld us (%ld ms)\n", down_time, down_time/1000);

  /*
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
  */

  /****************************************************************************************/




  //結果を表示したい場合はここ
  //***************************************************************************************

  printf("the number of total match tuple = %d\n",total);


  /*
  for(i=0;i<arg_right;i++){
    printf("%d: %8d%8d\n",i,Tright[i].id,Tright[i].val);

  }

  for(i=0;i<arg_left;i++){
    printf("%d: %8d%8d\n",i,Tleft[i].id,Tleft[i].val);

    }*/


  //ループが何番目かとleftテーブルの最後の値、join後のテーブルのidと最後の値を表示する
  /*
  for(i=0;i<300;i++){
    if(i%100000==0&&Tjoin[i].lval[0]>0){//if(Tjoin[j * arg_left + i].lval>0){
      printf("%d:left :join: %8d%8d%8d\n",i,Tjoin[i].id, Tjoin[i].lval[VAL_NUM-1], Tjoin[i].rval[VAL_NUM-1]);
    }
  }
  */



  for(i=0;i<3&&i<JT_SIZE;i++){
    printf("[%d]:left %8d \t:right: %8d\n",i,Tjoin[i].lid,Tjoin[i].rid);
    for(j = 0;j<VAL_NUM;j++){
      printf("join[%d]:left = %8d\tright = %8d\n",j,Tjoin[i].lval[j],Tjoin[i].rval[j]);
    }
    printf("\n");

  }

  /*****************************************************************************************/
  



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

  //割り当てたメモリを開放する
  tuple_free();

}


int
main(int argc, char *argv[])
{


  if(argc<4){
    if(argv[1]==NULL){
      printf("argument2 is nothing.\n");
      exit(1);
    }else{
      arg_left=atoi(argv[1]);
      printf("left num :%d\n",arg_left);
    }
    if(argv[2]==NULL){
      printf("argument1 is nothing.\n");
      exit(1);
    }else{
      arg_right=atoi(argv[2]);
      printf("right num :%d\n",arg_right);
    }    

  }


  //joinの実行
  join();


  return 0;
}
