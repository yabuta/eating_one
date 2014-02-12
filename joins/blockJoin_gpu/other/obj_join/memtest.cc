#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include "tuple_memtest.h"

TUPLE *Tright;
TUPLE *Tleft;
JOIN_TUPLE *Tjoin;

class Tuple
{
public:

  Tuple(){
    m_data[]=new char[50];
  }

  ~Tuple(){
    delete [] m_data;

  }

  String getValue(int idx){

    NValue value;
    return value.getNVlaueCopy(data);


  }

private:

  char *m_data;


};

class NValue{

public:
  
  String getNValue(char *data){

    String *res;
    memcpy(res,data,10);
    return res;

  }


private:


};

class Iterator{

public:

  void next(){

    

  }


private:


};





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
  CUdeviceptr lt_dev, rt_dev, p_dev,test_dev;
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


  gettimeofday(&tv_cal_s, NULL);

  /* block_x * block_y is decided by BLOCK_SIZE. */

  /*注意！
   *LEFTをx軸、RIGHTをy軸にした。ほかの場所と逆になっているので余裕があったら修正する
   *
   */


  /*block_x = MAX_LEFT < BLOCK_SIZE_X ? MAX_LEFT : BLOCK_SIZE_X;
  block_y = MAX_RIGHT < BLOCK_SIZE_Y ? MAX_RIGHT : BLOCK_SIZE_Y;
  grid_x = MAX_LEFT / block_x;
  if (MAX_LEFT % block_x != 0)
    grid_x++;
  grid_y = MAX_RIGHT / block_y;
  if (MAX_RIGHT % block_y != 0)
    grid_y++;
  */

  block_x=1;
  block_y=1;
  grid_x=1;
  grid_y=1;


  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *
   */



  //test用。配列の途中のポインタのGPUのアドレスを取得できるかどうか
  sprintf(fname, "%s/memtest.cubin", path);
  res = cuModuleLoad(&module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&function, module, "test");
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

  char *string;

  res = cuMemHostAlloc((void**)&string,10 * sizeof(char),CU_MEMHOSTALLOC_DEVICEMAP);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  for(i=0;i<10;i++){
    string[i]='a';
  }

  string[0]='t';
  string[1]='e';
  string[2]='s';
  string[3]='t';
  

  cuMemHostGetDevicePointer(&test_dev,string+2*sizeof(char),0);

  void *test_args[]={
    (void *)&test_dev
  };


  cuLaunchKernel(function,grid_x,grid_y,1,block_x,block_y,1,0,NULL,test_args,NULL);

  cuCtxSynchronize();
  

  res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload1 failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  
  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  //////////////////////////////////////////////////////////////////////////////////////
  //testここまで




  cuMemFreeHost(string);
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
