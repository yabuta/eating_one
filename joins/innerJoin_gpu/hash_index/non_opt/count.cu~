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

  
  if(x < left){
    int idx = lt[x].val % NB_BKT_ENT;
    /*
    int idx_c = idxcount[idx];
    int buck_a = buck_array[idx];
    uint temp = 0;
    */
    count[x]=0;
    for(int k = 0; k < idxcount[idx]; k++){
      if(bucket[buck_array[idx] + k].val == lt[x].val){
        count[x]++;
      }    
    }

  }
  if(x == left-1){
    count[x+1] = 0;
  }

}

}
