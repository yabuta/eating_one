/*
count the number of tuple matching criteria for join

*/

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"


extern "C" {

__global__
void count(
          TUPLE *lt,
          uint *count,
          BUCKET *bucket,
          int *buck_array,
          int *idxcount,
          int left
          ) 

{
    
  //i,jの方向を間違えないように
  /*
   *x軸が縦の方向、y軸が横の方向だよ。
   *だから、xがleft、yがrightに対応しているよ
   */

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(i < left){
    int idx = lt[i].val % NB_BKT_ENT;
    int idx_c = idxcount[idx];
    int buck_a = buck_array[idx];
    uint temp = 0;

    for(int k = 0; k < idx_c; k++){
      if(bucket[buck_a + k].val == lt[i].val){
        temp++;
      }    
    }

    count[i] = temp;

  }

}

}
