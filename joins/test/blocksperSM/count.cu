/*
count the number of tuple matching criteria for join

*/

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

/**
execution time in case jt[x]=lt[0] is faster than in case jt[x]=lt[x]


 **/

extern "C" {

__global__
void count(
           int *lt,
           int *jt,
           int left
          ) 

{

  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ int temp[1024];

  int temp2 = lt[x];

  if(x<left){
    for(uint i=0; i<256 ; i++){
      temp2=temp2%256;
      jt[x] += temp2;
    }
  }

  //jt[x] = lt[0];

}

}
