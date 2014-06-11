#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {

__global__
void count(
          TUPLE *lt,
          TUPLE *rt,
          int *count,
          int ltn,
          int rtn
          ) 

{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  
  /*
    transport tuple data to shared memory from global memory
   */

  __shared__ TUPLE Tright[BLOCK_SIZE_Y];
  for(uint j=0;threadIdx.x+j*BLOCK_SIZE_X<BLOCK_SIZE_Y&&(threadIdx.x+j*BLOCK_SIZE_X+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
    Tright[threadIdx.x + j*BLOCK_SIZE_X] = rt[threadIdx.x + j*BLOCK_SIZE_X + BLOCK_SIZE_Y * blockIdx.y];
  }

  __syncthreads();  

  TUPLE Tleft = lt[i];

  /*
    count loop
   */
  int ltn_g = ltn;
  int rtn_g = rtn;
  uint mcount = 0;

  if(i<ltn_g){
    for(uint j = 0; j<BLOCK_SIZE_Y &&((j+BLOCK_SIZE_Y*blockIdx.y)<rtn_g);j++){
      if((Tright[j].val==Tleft.val)) {
        mcount++;
      }
    }
    count[i + k] = mcount;  
    if(i+k == blockDim.x*gridDim.x*gridDim.y){
      count[i+k+1] = mcount;
    }
  }    

}


__global__ void join(
          TUPLE *lt,
          TUPLE *rt,
          JOIN_TUPLE *p,
          int *count,
          int ltn,
          int rtn
          ) 

{

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ TUPLE Tright[BLOCK_SIZE_Y];
  for(uint j=0;threadIdx.x+j*BLOCK_SIZE_X<BLOCK_SIZE_Y&&(threadIdx.x+j*BLOCK_SIZE_X+BLOCK_SIZE_Y*blockIdx.y)<rtn;j++){
    Tright[threadIdx.x + j*BLOCK_SIZE_X] = rt[threadIdx.x + j*BLOCK_SIZE_X + BLOCK_SIZE_Y * blockIdx.y];
  }
  __syncthreads();  


  TUPLE Tleft = lt[i];

  //the first write location

  int writeloc = 0;
  if(i != 0){
    writeloc = count[i + blockIdx.y*blockDim.x*gridDim.x];
  }
  int ltn_g = ltn;
  int rtn_g = rtn;

  if(i<ltn_g){
    for(uint j = 0; j<BLOCK_SIZE_Y &&((j+BLOCK_SIZE_Y*blockIdx.y)<rtn_g);j++){
 
      if((Tright[j].val==Tleft.val)) {

        p[writeloc].rid = Tright[j].id;
        p[writeloc].rval = Tright[j].val;
        p[writeloc].lid = Tleft.val;        
        p[writeloc].lval = Tleft.id;        
        writeloc++;
        
      }
    }
  }    
    
}    

}
