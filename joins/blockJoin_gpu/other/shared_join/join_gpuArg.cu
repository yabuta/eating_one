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
  //  int n = j * (*rtn) + i;

  

  __shared__ TUPLE Tleft[BLOCK_SIZE_X];
  if(threadIdx.y==0){
    for(j=0;(j<BLOCK_SIZE_X)&&((j+BLOCK_SIZE_X*blockIdx.x)<ltn);j++){
      Tleft[j] = lt[j + BLOCK_SIZE_X * blockIdx.x];
      //printf("%d %d %d\n",Tleft[j].id,j,blockIdx.x);
    }

  }

  __syncthreads();  

    /*
  for(j = BLOCK_SIZE_X * blockIdx.x + threadIdx.y;j < BLOCK_SIZE_X*(blockIdx.x+1)&&(j<ltn) ; j=j+BLOCK_SIZE_Y){
    Tleft[j] = lt[j];
    printf("%d %d %d\n",lt[j].id,j,blockIdx.x);
  }
    */

  TUPLE Tright = rt[i];


  for(j = 0; j<BLOCK_SIZE_X && ((j+BLOCK_SIZE_X*blockIdx.x)<ltn);j++){


    if(i<rtn){
      int n = (j + blockIdx.x*BLOCK_SIZE_X) * rtn + i;
    
      if(&(Tleft[j])==NULL||&(Tright)==NULL||&(p[n])==NULL){
        printf("memory error in .cu.\n");
        return;// -1;
      }
 
      //  if((lt[j].val[0]==rt[i].val[0])&&(j<*ltn)&&(i<*rtn)) {
      if((Tleft[j].val[0]==Tright.val[0])) {
        
        //時間計測はできないそうだよ。タイムスタンプをどうするかな

    
        for(k=0; k<VAL_NUM; k++) {
          p[n].lval[k] = Tleft[j].val[k];
          p[n].rval[k] = Tright.val[k];  
        }

        // lid & rid are just for debug
        p[n].lid = Tleft[j].id;
        p[n].rid = Tright.id;

      }
    }
  }    
    
}    

}
