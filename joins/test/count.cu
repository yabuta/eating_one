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
           int *lt,
           int *jt,
           int left
          ) 

{

  uint x = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ int temp[32];
  /*
  for(uint i=0 ;i<1024;i++){
    temp[i] = 0;
  }
  */
  /*
  for(uint i=0 ;i<1024;i++){
    temp[threadIdx.x]++;
  }
  */

  for(uint i=0 ;i<1024;i++){
    jt[x]++;
  }

  /*
  if(threadIdx.x==0&&blockIdx.x==0){
    for(uint i=0 ;i<1024;i++){
      printf("%d\n",temp[i]);
    }
  }
  */
  /*
  for(uint i=threadIdx.x ;i<32*1024;i+=blockDim.x){
    temp[i] = 0;
  }
  */
  __syncthreads();

}

}
