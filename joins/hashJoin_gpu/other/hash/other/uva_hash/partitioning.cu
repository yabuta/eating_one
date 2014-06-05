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
          int rows_num,
          int table_type
          ) 

{

  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;

  int PER_TH = LEFT_PER_TH;
  if(table_type != LEFT) PER_TH = RIGHT_PER_TH;

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

__global__
void partitioning(
          TUPLE *t,
          TUPLE *pt,
          int *L,
          int p_num,
          int t_num,
          int rows_num,
          int table_type
          ) 

{

  int p_n = p_num;
  int t_n = t_num;
  int rows_n = rows_num;

  int PER_TH = LEFT_PER_TH;
  if(table_type != LEFT) PER_TH = RIGHT_PER_TH;

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
  int temp = 0;
  if(x < t_n){
    for(int i = 0; i<PER_TH&&(DEF+threadIdx.x+i*Dim)<rows_n;i++){
      hash = t[DEF + threadIdx.x + i*Dim].val%p_n;
      temp = L[hash*t_n + x];

      pt[temp].key = t[DEF + threadIdx.x + i*Dim].key;  
      pt[temp].val = t[DEF + threadIdx.x + i*Dim].val;  
      L[hash*t_n + x] = temp + 1;
    }    
  
  }


}

}
