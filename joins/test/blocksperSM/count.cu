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

  jt[x] = lt[x];

  //jt[x] = lt[0];

}

}
