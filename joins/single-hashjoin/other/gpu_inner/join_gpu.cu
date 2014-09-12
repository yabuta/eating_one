#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {
__global__ void join(
          TUPLE *lt,
          TUPLE *rt,
          RESULT *jt,
          int *count,
          BUCKET *bucket,
          int *buck_array,
          int *idxcount,
          int left,
          int right
          ) 
{


  int x = threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;


  int writeloc = 0;
  if(y!=0){
    writeloc = count[y-1];
  }

  if(y < right){
    int idx = rt[y].val % NB_BKT_ENT;
    if(buck_array[idx] != -1 && x < idxcount[idx]){
      for(int i = x; i < idxcount[idx] ; i = i + BLOCK_SIZE_X){
        if(bucket[buck_array[idx] + i].val == rt[y].val){
          jt[writeloc + i].rkey = rt[y].key;
          jt[writeloc + i].rval = rt[y].val;
          jt[writeloc + i].lkey = lt[bucket[buck_array[idx] + i].adr].key;
          jt[writeloc + i].lval = lt[bucket[buck_array[idx] + i].adr].val;
          
          //printf("%d %d\n",jt[count[i] + k].rkey,jt[count[i] + k].lkey);
        }
      }
    }
  }
    
}    

}
