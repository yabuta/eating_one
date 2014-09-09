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
  __shared__ TUPLE sub_lt[JOIN_SHARED];

  for(int i=l_p[blockIdx.x] + threadIdx.x,j=threadIdx.x; i<l_p[blockIdx.x+1]; i += blockDim.x, j += blockDim.x){
    if(j<JOIN_SHARED){
      sub_lt[j] = lt[i];
    }

  }
  __syncthreads();

  /*
  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<r_p[radix[blockIdx.x]+1] ; k += blockDim.x){
    for(int i=0; i<l_p[blockIdx.x+1]-l_p[blockIdx.x] ;i++){
      if(sub_lt[i].val == rt[k].val){
        count[x]++;
      }
    }
  }
  */
  int temp=0;
  int temp2 = r_p[radix[blockIdx.x]+1];
  int temp3 = l_p[blockIdx.x+1] - l_p[blockIdx.x];
  int count_x_temp = 0;

  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<temp2 ; k += blockDim.x){
    temp = rt[k].val;
    for(int i=0; i<temp3 ;i++){
      if(sub_lt[i].val == temp){
        count_x_temp++;
      }
    }
  }
  count[x] = count_x_temp;



  if(x == gridDim.x*blockDim.x-1){
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

  __shared__ TUPLE sub_lt[JOIN_SHARED];

  for(int i=l_p[blockIdx.x] + threadIdx.x,j=threadIdx.x; i<l_p[blockIdx.x+1]; i += blockDim.x, j += blockDim.x){
    if(j<JOIN_SHARED){
      sub_lt[j].key = lt[i].key;
      sub_lt[j].val = lt[i].val;
    }
  }
  __syncthreads();

  TUPLE temp;
  int temp2 = r_p[radix[blockIdx.x]+1];
  int temp3 = l_p[blockIdx.x+1] - l_p[blockIdx.x];
  int tcount=count[x];

  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<temp2 ; k += blockDim.x){
    temp.key = rt[k].key;
    temp.val = rt[k].val;
    for(int i=0; i<temp3 ;i++){
      if(sub_lt[i].val == temp.val){
        jt[tcount].rkey = temp.key;
        jt[tcount].rval = temp.val;
        jt[tcount].lkey = sub_lt[i].key;
        jt[tcount].lval = sub_lt[i].val;
        tcount++;
      }
    }
  }
    
}    

}
