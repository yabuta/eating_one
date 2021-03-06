#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {
__global__ void join(
          TUPLE *lt,
          TUPLE *rt,
          JOIN_TUPLE *p,
          int *count,
          int ltn,
          int rtn
          ) 

{

  int j,k;
    
  //i,jの方向を間違えないように
  /*
   *x軸が縦の方向、y軸が横の方向だよ。
   *だから、xがleft、yがrightに対応しているよ
   */

  //int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ TUPLE Tleft[BLOCK_SIZE_X];
  if(threadIdx.y==0){
    for(j=0;(j<BLOCK_SIZE_X)&&((j+BLOCK_SIZE_X*blockIdx.x)<ltn);j++){
      Tleft[j] = lt[j + BLOCK_SIZE_X * blockIdx.x];
    }

  }

  __syncthreads();  

  TUPLE Tright = rt[i];

  //the first write location

  int writeloc = 0;
  if(i != 0){
    writeloc = count[i + blockIdx.x*blockDim.y*gridDim.y -1];
  }
  int ltn_g = ltn;
  int rtn_g = rtn;


  for(j = 0; j<BLOCK_SIZE_X &&((j+BLOCK_SIZE_X*blockIdx.x)<ltn_g);j++){

    if(i<rtn_g){
    
      if(&(Tleft[j])==NULL||&(Tright)==NULL||&(p[writeloc])==NULL){
        printf("memory error in .cu.\n");
        return;// -1;
      }
 
      if((Tleft[j].val[0]==Tright.val[0])) {
        
        for(k=0; k<VAL_NUM; k++) {
          p[writeloc].lval[k] = Tleft[j].val[k];
          p[writeloc].rval[k] = Tright.val[k];  
        }

        // lid & rid are just for debug
        p[writeloc].lid = Tleft[j].id;
        p[writeloc].rid = Tright.id;
        
        writeloc++;
        
      }
    }
  }    
    
}    

}
