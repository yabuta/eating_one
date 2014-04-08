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

  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;

  int DEF = blockIdx.x * blockDim.x * PER_TH;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int Dim = 0;
  if(gridDim.x-1 == blockIdx.x){
    Dim = t_n - blockIdx.x*blockDim.x;
  }else{
    Dim = blockDim.x;
  }


  // Matching phase
  int hash = 0;

  if(x < t_n){
    for(int i = 0; i<PER_TH&&(DEF+threadIdx.x+i*Dim)<rows_n;i++){
      hash = t[DEF + threadIdx.x + i*Dim].val % p_n;
      L[hash*t_n + x]++;
    }
  }

}
}
