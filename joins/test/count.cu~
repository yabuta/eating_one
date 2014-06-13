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
           RESULT *jt,
           int left

          /*
          uint *count,
          BUCKET *bucket,
          int *buck_array,
          int *idxcount,
          int left
          */
          ) 

{

  uint i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < left){
    jt[i].lkey = 1;
    jt[i].lval = 1;
    jt[i].rkey = 0;
    jt[i].rval = 0;
  }

  /*
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(x < left){
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
  */
}

}
