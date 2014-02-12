/*
count the number of match tuple in each partition and each thread

*/

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"


extern "C" {

__global__
void count_partitioning(
          TUPLE *t,
          int *L,
          int p,
          int t_num
          ) 

{

  int x = blockIdx.x * blockDim.x + threadIdx.x;

  // Matching phase
  int hash = 0;
  for(int i = 0; i<PER_TH;i++){
    hash = t[x*PER_TH + i].val%p;
    L[hash*t_num + x]++;  
  }

}

}
