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

  int PER_TH = (table_type==LEFT) ? LEFT_PER_TH:RIGHT_PER_TH;
  int DEF = blockIdx.x * blockDim.x * PER_TH;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int Dim = (gridDim.x-1==blockIdx.x) ? (t_n-blockIdx.x*blockDim.x):blockDim.x;


  // Matching phase
  int hash = 0;

  /*
  if(x < t_n){
    for(int i = 0; i<PER_TH&&(DEF+PER_TH*threadIdx.x+i)<rows_n;i++){
      hash = t[DEF + PER_TH*threadIdx.x + i].val % p_n;
      L[hash*t_n + x]++;
    }
  }
  */

  if(x < t_n){
    for(int i = threadIdx.x; i<PER_TH*Dim&&(DEF+i)<rows_n;i+=Dim){
      hash = t[DEF + i].val % p_n;
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

  int PER_TH = (table_type==LEFT) ? LEFT_PER_TH:RIGHT_PER_TH;
  int DEF = blockIdx.x * blockDim.x * PER_TH;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int Dim = (gridDim.x-1==blockIdx.x) ? (t_n-blockIdx.x*blockDim.x):blockDim.x;

  // Matching phase
  int hash = 0;
  int temp = 0;
  TUPLE tt;

  /*
  if(x < t_n){
    for(int i = 0; i<PER_TH&&(DEF+threadIdx.x*PER_TH+i)<rows_n;i++){
      tt=t[DEF + threadIdx.x*PER_TH + i];
      hash = tt.val%p_n;
      temp = L[hash*t_n + x]++;
      pt[temp] = tt;  
    }    
  }
  */

  if(x < t_n){
    for(int i = threadIdx.x; i<PER_TH*Dim&&(DEF+i)<rows_n;i+=Dim){
      tt = t[DEF+i];
      hash = tt.val%p_n;
      temp = L[hash*t_n + x]++;

      pt[temp] = tt;  
    }    
  
  }


}

}
