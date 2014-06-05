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
          int *lp,
          int right,
          int left
          )

{

  int x = blockIdx.x*blockDim.x*gridDim.y + blockDim.x*blockIdx.y + threadIdx.x;

  //insert partition left table in shared memory
  __shared__ TUPLE sub_lt[B_ROW_NUM];

  for(int i=lp[blockIdx.x] + threadIdx.x,j=threadIdx.x; i<lp[blockIdx.x+1]; i += blockDim.x, j += blockDim.x){
    if(j<B_ROW_NUM){
      sub_lt[j].key = lt[i].key;
      sub_lt[j].val = lt[i].val;
    }
  }

  __syncthreads();

  int temp=0;
  int temp2 = r_p[radix[blockIdx.x]+1];
  int temp3 = lp[blockIdx.x+1] - lp[blockIdx.x];
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

}

__global__ void join(
          TUPLE *lt,
          TUPLE *rt,
          RESULT *jt,
          int *count,
          int *r_p,
          int *radix,
          int *lp,
          int right,
          int left
          ) 
{

  //int x = blockIdx.x*blockDim.x + threadIdx.x;
  int x = blockIdx.x*blockDim.x*gridDim.y + blockDim.x*blockIdx.y + threadIdx.x;

  __shared__ TUPLE sub_lt[B_ROW_NUM];

  for(int i=lp[blockIdx.x] + threadIdx.x,j=threadIdx.x; i<lp[blockIdx.x+1]; i += blockDim.x, j += blockDim.x){
    if(j<B_ROW_NUM){
      sub_lt[j].key = lt[i].key;
      sub_lt[j].val = lt[i].val;
    }
  }


  __syncthreads();

  /*
  int x_limit = 0;

  if(r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]]%GRID_SIZE_Y == 0){
    x_limit = (r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]])/GRID_SIZE_Y;
  }else{
    x_limit = (r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]])/GRID_SIZE_Y + 1;
  }
  */
  
  TUPLE temp;
  uint temp2 = r_p[radix[blockIdx.x]+1];
  uint temp3 = lp[blockIdx.x+1] - lp[blockIdx.x];
  uint tcount = count[x];

  for(uint k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<temp2 ; k += blockDim.x){
    temp.key = rt[k].key;
    temp.val = rt[k].val;
    for(uint i=0; i<temp3 ;i++){
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
