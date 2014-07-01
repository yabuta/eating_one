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
          int *r_p,
          int *radix,
          int *l_p,
          int right,
          int left
          )

{

  int x = blockIdx.x*blockDim.x*gridDim.y + blockDim.x*blockIdx.y + threadIdx.x;

  //insert partition left table in shared memory

  count[x] = 0;
  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<r_p[radix[blockIdx.x]+1] ; k+=blockDim.x){
    for(int i=l_p[blockIdx.x]; i<l_p[blockIdx.x+1] ;i++){
      if(lt[i].val == rt[k].val){
        count[x]++;
      }
    }
  }

  if(x == blockIdx.x*gridDim.x-1){
    count[x+1] = 0;
  }


}

__global__ void join(
          TUPLE *lt,
          TUPLE *rt,
          RESULT *jt,
          int *count,
          int *r_p,
          int *radix,
          int *l_p,
          int right,
          int left
          ) 
{

  //int x = blockIdx.x*blockDim.x + threadIdx.x;
  int x = blockIdx.x*blockDim.x*gridDim.y + blockDim.x*blockIdx.y + threadIdx.x;

  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<r_p[radix[blockIdx.x]+1] ; k += blockDim.x){
    for(int i=l_p[blockIdx.x]; i<l_p[blockIdx.x+1] ;i++){
      if(lt[i].val == rt[k].val){
        jt[count[x]].rkey = rt[k].key;
        jt[count[x]].rval = rt[k].val;
        jt[count[x]].lkey = lt[i].key;
        jt[count[x]].lval = lt[i].val;

        count[x]++;

      }
    }
  }

    
}    

}
