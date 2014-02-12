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


  int y = blockIdx.y * blockDim.y + threadIdx.y;


  int writeloc = 0;
  if(y!=0){
    writeloc = count[y-1];
  }

  if(y < right){
    int idx = rt[y].val % NB_BKT_ENT;
    if(buck_array[idx] != -1){
      int i = 0;
      for(int k = 0; k < idxcount[idx]; k++){
        if(bucket[buck_array[idx] + k].val == rt[y].val){
          jt[writeloc + i].rkey = rt[y].key;
          jt[writeloc + i].rval = rt[y].val;
          jt[writeloc + i].lkey = lt[bucket[buck_array[idx] + k].adr].key;
          jt[writeloc + i].lval = lt[bucket[buck_array[idx] + k].adr].val;
          
          i++;
          //printf("%d %d\n",jt[count[i] + k].rkey,jt[count[i] + k].lkey);
        }
      }
    }
  }



  //shared memory experience
  /*
  __shared__ int ba[NB_BKT_ENT];

  for(int i=0; i<NB_BKT_ENT ;i++){
    ba[i] = buck_array[i];
  }

  __syncthreads();

  int writeloc = 0;
  if(y!=0){
    writeloc = count[y-1];
  }

  if(y < right){
    int idx = rt[y].val % NB_BKT_ENT;
    if(ba[idx] != -1){
      int i = 0;
      for(int k = 0; k < idxcount[idx]; k++){
        if(bucket[ba[idx] + k].val == rt[y].val){
          jt[writeloc + i].rkey = rt[y].key;
          jt[writeloc + i].rval = rt[y].val;
          jt[writeloc + i].lkey = lt[bucket[ba[idx] + k].adr].key;
          jt[writeloc + i].lval = lt[bucket[ba[idx] + k].adr].val;
          
          i++;
          //printf("%d %d\n",jt[count[i] + k].rkey,jt[count[i] + k].lkey);
        }
      }
    }
  }

  */
    
}    

}
