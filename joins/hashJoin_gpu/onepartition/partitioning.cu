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
void lcount_partitioning(
          TUPLE *t,
          uint *L,
          int p_num,
          int t_num,
          int rows_num,
          int loop
          ) 

{

  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;

  int x = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int part[SHARED_MAX];

  for(uint i=threadIdx.x; i<SHARED_MAX ; i+=blockDim.x){
    part[i] = 0;
  }
  __syncthreads();

  //int PER_TH = (table_type==LEFT) ? LEFT_PER_TH:RIGHT_PER_TH;

  int DEF = blockIdx.x * blockDim.x * LEFT_PER_TH;
  int Dim = (gridDim.x-1 == blockIdx.x) ? (t_n - blockIdx.x*blockDim.x):blockDim.x;
  // Matching phase
  int hash = 0;

  if(x < t_n){

    for(uint i=0; i<LEFT_PER_TH&&(DEF+threadIdx.x*LEFT_PER_TH+i)<rows_n; i++){
      hash = loop==0 ? (t[DEF+threadIdx.x*LEFT_PER_TH+i].val) % p_n:
        (t[DEF+threadIdx.x*LEFT_PER_TH+i].val>>(RADIX*loop)) % p_n;
      part[hash*Dim + threadIdx.x]++;
      
    } 
    for(uint j=0 ; j*Dim+threadIdx.x<p_n*Dim ; j++){
      L[t_n*j + blockIdx.x*Dim + threadIdx.x] = part[j*Dim+threadIdx.x];
    }
  }
}

__global__
void lpartitioning(
          TUPLE *t,
          TUPLE *pt,
          uint *L,
          int p_num,
          int t_num,
          int rows_num,
          int loop
          ) 

{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;


  int DEF = blockIdx.x * blockDim.x * LEFT_PER_TH;
  int Dim = (gridDim.x-1 == blockIdx.x) ? (t_n - blockIdx.x*blockDim.x):blockDim.x;

  __shared__ int part[SHARED_MAX];
  for(uint j=0 ; j*blockDim.x+threadIdx.x<p_n*blockDim.x ; j++){
    part[j*blockDim.x+threadIdx.x] = L[t_n*j+blockIdx.x*Dim+threadIdx.x];
  }
  
  __syncthreads();

  // Matching phase
  int hash = 0;
  int temp = 0;
  TUPLE tt;

  if(x < t_n){
    for(uint i=0; i<LEFT_PER_TH&&(DEF+threadIdx.x*LEFT_PER_TH+i)<rows_n; i++){
        tt = t[DEF+threadIdx.x*LEFT_PER_TH+i];
        hash = loop==0 ? (t[DEF+threadIdx.x*LEFT_PER_TH+i].val) % p_n:
          (t[DEF+threadIdx.x*LEFT_PER_TH+i].val>>(RADIX*loop)) % p_n;

        temp = part[hash*Dim + threadIdx.x]++;
        pt[temp] = tt; 

    } 
  }

}

__global__
void rcount_partitioning(
          TUPLE *t,
          uint *L,
          int p_num,
          int t_num,
          int rows_num,
          int loop
          ) 

{

  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;

  int x = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int part[SHARED_MAX];

  for(uint i=threadIdx.x; i<SHARED_MAX ; i+=blockDim.x){
    part[i] = 0;
  }
  __syncthreads();

  //int PER_TH = (table_type==LEFT) ? LEFT_PER_TH:RIGHT_PER_TH;

  int DEF = blockIdx.x * blockDim.x * RIGHT_PER_TH;
  int Dim = (gridDim.x-1 == blockIdx.x) ? (t_n - blockIdx.x*blockDim.x):blockDim.x;
  // Matching phase
  int hash = 0;

  if(x < t_n){

    for(uint i=0; i<RIGHT_PER_TH&&(DEF+threadIdx.x*RIGHT_PER_TH+i)<rows_n; i++){
      //hash = (i>>(8*loop)) % p_n;
      //hash = (t[DEF+threadIdx.x*RIGHT_PER_TH+i].val>>(radix*loop)) % p_n;
      hash = loop==0 ? (t[DEF+threadIdx.x*LEFT_PER_TH+i].val) % p_n:
        (t[DEF+threadIdx.x*LEFT_PER_TH+i].val>>(RADIX*loop)) % p_n;

      part[hash*Dim + threadIdx.x]++;
      
    } 
    for(uint j=0 ; j*Dim+threadIdx.x<p_n*Dim ; j++){
      L[t_n*j + blockIdx.x*Dim + threadIdx.x] = part[j*Dim+threadIdx.x];
    }
  }
}


__global__
void rpartitioning(
          TUPLE *t,
          TUPLE *pt,
          uint *L,
          int p_num,
          int t_num,
          int rows_num,
          int loop
          ) 

{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;


  //int PER_TH = (table_type==LEFT) ? LEFT_PER_TH:RIGHT_PER_TH;
  int DEF = blockIdx.x * blockDim.x * RIGHT_PER_TH;
  int Dim = (gridDim.x-1 == blockIdx.x) ? (t_n - blockIdx.x*blockDim.x):blockDim.x;

  __shared__ int part[SHARED_MAX];
  for(uint j=0 ; j*blockDim.x+threadIdx.x<p_n*blockDim.x ; j++){
    part[j*blockDim.x+threadIdx.x]=L[t_n*j+blockIdx.x*blockDim.x+threadIdx.x];
  }
  
  __syncthreads();

  // Matching phase
  int hash = 0;
  int temp = 0;
  TUPLE tt;

  if(x < t_n){

    for(uint i=0; i<RIGHT_PER_TH&&(DEF+threadIdx.x*RIGHT_PER_TH+i)<rows_n; i++){
        tt = t[DEF+threadIdx.x*RIGHT_PER_TH+i];
        //hash = (tt.val>>loop*radix) % p_n;
        hash = loop==0 ? (t[DEF+threadIdx.x*LEFT_PER_TH+i].val) %p_n
          :(t[DEF+threadIdx.x*LEFT_PER_TH+i].val>>(RADIX*loop)) % p_n;
        temp = part[hash*Dim + threadIdx.x]++;
        pt[temp] = tt; 

    } 
  }

}

__global__
void countPartition(
          TUPLE *t,
          uint *startpos,
          int p_num,
          int rows_num
          ) 
{

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  if(x<rows_num){
    int p = t[x].val%p_num;
    atomicAdd(&(startpos[p]),1);
  }

  if(x==blockIdx.x*blockDim.x-1){
    startpos[x+1]=0;
  }


}



}
