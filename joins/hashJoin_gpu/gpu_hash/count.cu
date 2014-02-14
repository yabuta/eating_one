/*
count the number of tuple matching criteria for join
block_x_size 128
block_y_size 64
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

  int x = blockIdx.x*blockDim.x + threadIdx.x;

  //insert partition left table in shared memory
  __shared__ TUPLE sub_lt[B_ROW_NUM];

  //printf("%d\t%d\n",lp[blockIdx.x+1],lp[blockIdx.x]);

  for(int i=lp[blockIdx.x] + threadIdx.x,j=threadIdx.x; i<lp[blockIdx.x+1]; i += BLOCK_SIZE_X, j += BLOCK_SIZE_X){
    if(j<B_ROW_NUM){
      sub_lt[j].key = lt[i].key;
      sub_lt[j].val = lt[i].val;
    }
  }

  /*
  if(threadIdx.x==0){
    for(int j=0; j<lp[blockIdx.x+1]-lp[blockIdx.x]; j++){
      if(j<B_ROW_NUM){
        sub_lt[j].key = lt[j+lp[blockIdx.x]].key;
        sub_lt[j].val = lt[j+lp[blockIdx.x]].val;
      }
    }
  }
  */

  /*
  if(threadIdx.x<lp[blockIdx.x+1]-lp[blockIdx.x]){
    sub_lt[threadIdx.x].key = lt[threadIdx.x+lp[blockIdx.x]].key;
    sub_lt[threadIdx.x].val = lt[threadIdx.x+lp[blockIdx.x]].val;
  }
  */


  __syncthreads();


  /*
  if(blockIdx.x==2){
    for(int i=0 ; i<lp[blockIdx.x+1] - lp[blockIdx.x] ; i++){
      printf("thread = %d\t%d\n",threadIdx.x,sub_lt[i].val);
    }
  }
  */

  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<r_p[radix[blockIdx.x]+1] ; k += BLOCK_SIZE_X){
    for(int i=0; i<lp[blockIdx.x+1] - lp[blockIdx.x] ;i++){
      if(sub_lt[i].val == rt[k].val){
        count[x]++;
      }
    }
  }

  //printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",temp,temp2,blockIdx.x,threadIdx.x,lp[blockIdx.x+1],lp[blockIdx.x],r_p[radix[blockIdx.x]+1],r_p[radix[blockIdx.x]]);


}

}
