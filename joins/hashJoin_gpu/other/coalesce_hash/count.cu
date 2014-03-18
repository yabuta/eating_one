/*
count the number of tuple matching criteria for join
block_x_size 128
block_y_size 1
grid_x_size new_p_num
grid_y_size 1

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
          TUPLE *rt,
          int *count,
          int *r_p,
          int *radix,
          int *lp,
          int right,
          int left
          )

{

  //int x = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x + threadIdx.y*blockDim.x;
  int x = blockIdx.y*blockDim.x*blockDim.y*gridDim.x + blockDim.x*blockDim.y*blockIdx.x + threadIdx.y*blockDim.x + threadIdx.x;

  __shared__ TUPLE sub_lt[J_T_LEFT];

  for(int i=lp[blockIdx.x] + threadIdx.x,j=threadIdx.x; i<lp[blockIdx.x+1]; i += blockDim.x, j += blockDim.x){
    if(j<B_ROW_NUM){
      sub_lt[j].key = lt[i].key;
      sub_lt[j].val = lt[i].val;
    }
  }

  __syncthreads();

  int temp=0;
  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<r_p[radix[blockIdx.x]+1] ; k += blockDim.x){
    temp = rt[k].val;
    for(int i=0; i<lp[blockIdx.x+1]-lp[blockIdx.x] ;i++){
      if(sub_lt[i].val == temp){
        count[x]++;
      }

    }

  }

}

}
