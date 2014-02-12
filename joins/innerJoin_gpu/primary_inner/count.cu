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
          TUPLE *rt,
          int *count,
          BUCKET *bucket,
          int *buck_array,
          int *idxcount,
          int right
          ) 

{
    
  //i,jの方向を間違えないように
  /*
   *x軸が縦の方向、y軸が横の方向だよ。
   *だから、xがleft、yがrightに対応しているよ
   */

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(i < right){
    int idx = rt[i].val % NB_BKT_ENT;
    if(buck_array[idx] != -1){
      for(int k = 0; k < idxcount[idx]; k++){
        if(bucket[buck_array[idx] + k].val == rt[i].val){
          count[i]++;
        }
      }
    }
  }

}

}
