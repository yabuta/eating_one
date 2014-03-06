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
void partitioning(
          TUPLE *t,
          TUPLE *pt,
          int *L,
          int p,
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
      hash = t[DEF + threadIdx.x + i*blockDim.x].val%p;
      pt[L[hash*t_num + x]].key = t[DEF + threadIdx.x + i*blockDim.x].key;  
      pt[L[hash*t_num + x]].val = t[DEF + threadIdx.x + i*blockDim.x].val;  
      L[hash*t_num + x]++;
      //printf("i = %d\tloc = %d\tt = %d\n",hash*t_num + x,L[hash*t_num + x],t[x*PER_TH + i].val);
      
    }
  }


}

}
