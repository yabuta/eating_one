#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
//#include "sharedmem.cuh"
#include "tuple.h"


extern "C" {
__global__


//main引数をとるかどうか

void count(
          TUPLE *lt,
          TUPLE *rt,
          int *count,
          int ltn,
          int rtn
          ) 

{
  extern __shared__ char sharedmem[];
  int sdata = (int *) sharedmem;

  int blocks = blockIdx.x * blockIdx.y;

  printf("%d %d\n",blocks,sdata);
  //i,jの方向を間違えないように
  /*
   *x軸が縦の方向、y軸が横の方向だよ。
   *だから、xがleft、yがrightに対応しているよ
   */

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  //  int n = j * (*rtn) + i;
  //int n = j * (rtn) + i;
  int k;

  if(&(lt[j])==NULL||&(rt[i])==NULL){
    printf("memory error in .cu.\n");
    return;// -1;
  }

  for(k = 0;k<blockDim.x * blockDim.y;k++){
    if(count[i]!=0){
      printf("count is not allzero.");
      return;
    }
  }
  if((lt[j].val[0]==rt[i].val[0])&&(j<ltn)&&(i<rtn)) {

    //条件に合致する場合、countを+1する。
    //if corresponding , count += 1 
    //sdata = count[blocks];
    sdata++;
    __syncthreads();
    count[blocks] = sdata;

    //printf("%d\n",count[0]);
    //printf("%d\n",blockIdx.x);
    //printf("%d\n",blockIdx.y);
    

  }    
    
}    

}
