#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {
__global__ void join(
          TUPLE *rt,
          TUPLE *lt,
          RESULT *jt,
          uint *count,
          BUCKET *bucket,
          int *buck_array,
          int *idxcount,
          int right,
          int left
          ) 
{


  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x < left){
    int idx = lt[x].val % NB_BKT_ENT;

    if(buck_array[idx] != -1){
      for(int k = 0; k < idxcount[idx]; k++){
        if(bucket[buck_array[idx] + k].val == lt[x].val){
          jt[count[x]].lkey = lt[x].key;
          jt[count[x]].lval = lt[x].val;
          jt[count[x]].rkey = rt[bucket[buck_array[idx] + k].adr].key;
          jt[count[x]].rval = rt[bucket[buck_array[idx] + k].adr].val;
          
          count[x]++;
        }
      }
    }
  }
    
}    

}
