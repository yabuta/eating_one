/*
count the number of tuple matching criteria for join

*/

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

  int j;
    
  //i,jの方向を間違えないように
  /*
   *x軸が縦の方向、y軸が横の方向だよ。
   *だから、xがleft、yがrightに対応しているよ
   */

  //i:the number of y element k:the number of one brefore x element * the total y element
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * gridDim.y * blockDim.y;

  
  /*
    transport tuple data to shared memory from global memory
   */

  __shared__ TUPLE Tleft[BLOCK_SIZE_X];
  if(threadIdx.y==0){
    for(j=0;(j<BLOCK_SIZE_X)&&((j+BLOCK_SIZE_X*blockIdx.x)<ltn);j++){
      Tleft[j] = lt[j + BLOCK_SIZE_X * blockIdx.x];
    }

  }

  __syncthreads();  

  TUPLE Tright = rt[i];

  /*
    count loop
   */
  int ltn_g = ltn;
  int rtn_g = rtn;


  for(j = 0; j<BLOCK_SIZE_X &&((j+BLOCK_SIZE_X*blockIdx.x)<ltn_g);j++){
    if(i<rtn_g){
     
      if(&(Tleft[j])==NULL||&(Tright)==NULL){
        printf("memory error in .cu.\n");
        return;// -1;
      }
 
      //tuple match or not
      if((Tleft[j].val[0]==Tright.val[0])) {

        count[i + k]++;

      }
    }
  }    

}

}
