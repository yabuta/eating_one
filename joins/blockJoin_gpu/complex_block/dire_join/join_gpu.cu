#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>
#include "tuple.h"

#define DISTANCE 1

extern "C" {

__device__
bool eval(TUPLE rt,TUPLE lt){

  //double dis = DISTANCE * DISTANCE;
  double temp = 0;
  double temp2 = 0;
  for(uint i = 0; i<VAL_NUM ; i++){
    temp2 = rt.val[i]-lt.val[i];
    temp += temp2 * temp2;
  }

  return temp < DISTANCE*DISTANCE;
  //return temp < DISTANCE * DISTANCE;
  //return rt.val[0]==lt.val[0];

}



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
      if(eval(Tright[j],Tleft)) {
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
 
      if(eval(Tright[j],Tleft)) {

        p[writeloc].rid = Tright[j].id;
        p[writeloc].lid = Tleft.id;
        for(uint valnum=0; valnum<VAL_NUM ; valnum++){
          p[writeloc].rval[valnum] = Tright[j].val[valnum];  
          p[writeloc].lval[valnum] = Tleft.val[valnum];                
        }
        writeloc++;
        
      }
    }
  }    
    
}    

}
