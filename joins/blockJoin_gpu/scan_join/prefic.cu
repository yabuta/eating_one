#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"


extern "C" {

__global__
void count(
          TUPLE *lt,
          TUPLE *rt,
          int *count,
          int ltn,
          int rtn
          ) 

{
  //i,jの方向を間違えないように
  /*
   *x軸が縦の方向、y軸が横の方向だよ。
   *だから、xがleft、yがrightに対応しているよ
   */
  
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int n = j * (rtn) + i;

  if(&(lt[j])==NULL||&(rt[i])==NULL){
    printf("memory error in .cu.\n");
    return;// -1;
  }

  
  if((lt[j].val[0]==rt[i].val[0])&&(i<rtn)&&(j<ltn)) {
      
    //条件に合致する場合、countを1にする。
    //if corresponding , count = 1 
     count[n] = 1;
   }
 
}

}
