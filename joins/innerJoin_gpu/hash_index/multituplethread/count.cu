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

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(TH_TUPLE*x < left){
    int val = lt[x].val;
    int idx = val % NB_BKT_ENT;
    int idx_c = idxcount[idx];
    int buck_a = buck_array[idx];
    uint temp = 0;

    for(int k = 0; k < idx_c; k++){
      if(bucket[buck_a + k].val == val){
        temp++;
      }    
    }
    count[x] = temp;

  }
  if(x == left-1){
    count[x+1] = 0;
  }

}

}
