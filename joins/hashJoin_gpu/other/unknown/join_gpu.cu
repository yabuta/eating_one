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

  int x_limit = 0;

  if(r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]]%GRID_SIZE_Y == 0){
    x_limit = (r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]])/GRID_SIZE_Y;
  }else{
    x_limit = (r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]])/GRID_SIZE_Y + 1;
  }

  
  TUPLE temp;
  int tcount=count[x];

  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x+x_limit*blockIdx.y ; k<r_p[radix[blockIdx.x]+1] && k < r_p[radix[blockIdx.x]] + x_limit*(blockIdx.y+1) ; k += blockDim.x){
    //for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<r_p[radix[blockIdx.x]+1] ; k += blockDim.x){
    temp.key = rt[k].key;
    temp.val = rt[k].val;
    for(int i=0; i<lp[blockIdx.x+1] - lp[blockIdx.x] ;i++){
      if(sub_lt[i].val == temp.val){

        jt[tcount].rkey = temp.key;
        jt[tcount].rval = temp.val;
        jt[tcount].lkey = sub_lt[i].key;
        jt[tcount].lval = sub_lt[i].val;

        /*
        temp.key = sub_lt[i].key;
        temp.val = sub_lt[i].val;
        temp.key = sub_lt[i].key;
        temp.val = sub_lt[i].val;
        */
        tcount++;

      }
    }
  }

    
}    

}
