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
          uint *L,
          int p_num,
          int t_num,
          int rows_num,
          int table_type
          ) 

{

  __shared__ uint part[SHARED_MAX];
  for(uint i=threadIdx.x; i<SHARED_MAX ; i+=blockDim.x){
    part[i] = 0;
  }
  __syncthreads();

  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;

  int PER_TH = (table_type==LEFT) ? LEFT_PER_TH:RIGHT_PER_TH;

  int DEF = blockIdx.x * blockDim.x * PER_TH;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int Dim = (gridDim.x-1 == blockIdx.x) ? (t_n - blockIdx.x*blockDim.x):blockDim.x;

  // Matching phase
  int hash = 0;

  if(x < t_n){
    for(int i = threadIdx.x; i<PER_TH*Dim&&(DEF+i)<rows_n;i+=Dim){
      hash = t[DEF + i].val % p_n;
      part[hash*blockDim.x + threadIdx.x]++;
    }
  }

  for(uint j=0 ; j*blockDim.x+threadIdx.x<p_n*blockDim.x ; j++){
    L[t_n*j + blockIdx.x*blockDim.x + threadIdx.x] = part[j*blockDim.x+threadIdx.x];
  }

}

__global__
void partitioning(
          TUPLE *t,
          TUPLE *pt,
          uint *L,
          int p_num,
          int t_num,
          int rows_num,
          int table_type
          ) 

{

  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;

  int PER_TH = (table_type==LEFT) ? LEFT_PER_TH:RIGHT_PER_TH;
  int DEF = blockIdx.x * blockDim.x * PER_TH;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int Dim = (gridDim.x-1 == blockIdx.x) ? (t_n - blockIdx.x*blockDim.x):blockDim.x;


  __shared__ uint part[SHARED_MAX];
  for(uint j=0 ; j*blockDim.x+threadIdx.x<p_n*blockDim.x ; j++){
    part[j*blockDim.x+threadIdx.x]=L[t_n*j+blockIdx.x*blockDim.x+threadIdx.x];
  }

  __syncthreads();


  // Matching phase
  int hash = 0;
  int temp = 0;
  TUPLE tt;

  if(x < t_n){
    for(int i = threadIdx.x; i<PER_TH*Dim&&(DEF+i)<rows_n;i+=Dim){
      tt = t[DEF+i];
      hash = tt.val % p_n;
      temp = part[hash*blockDim.x + threadIdx.x]++;

      pt[temp].key = tt.key;  
      pt[temp].val = tt.val;  
    }    

  }

}

}
