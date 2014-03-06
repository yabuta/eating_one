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
          int p_num,
          int t_num,
          int rows_num
          ) 

{

  int DEF = blockIdx.x * blockDim.x * PER_TH;
  int x = blockIdx.x * blockDim.x + threadIdx.x;


  // Matching phase
  int hash = 0;
  if(x < t_num){
    for(int i = 0; i<PER_TH&&(DEF+threadIdx.x+i*blockDim.x)<rows_num;i++){
      hash = t[DEF + threadIdx.x + i*blockDim.x].val % p_num;
      L[hash*t_num + x]++;

    }
  }

}
}
