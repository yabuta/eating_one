#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"


extern "C" {
__global__

void join(TUPLE *lt,TUPLE *rt,JOIN_TUPLE *p) {
    
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int n = j * MAX_LEFT + i;

  if(&(lt[i])==NULL||&(rt[j])==NULL||&(p[n])==NULL){
    printf("memory error in .cu.\n");
    return;// -1;
  }else if((lt[i].val==rt[j].val)&&(i<MAX_LEFT)&&(j<MAX_RIGHT)){
      
    //時間計測はできないそうだよ。タイムスタンプをどうするかな
    //gettimeofday(&p->t, NULL);

    p[n].lval = lt[i].val;
    p[n].rval = rt[j].val;  
    // lid & rid are just for debug
    p[n].lid = lt[i].id;
    p[n].rid = rt[j].id;
      
    if(i==0){
      //printf("Left Tuple %8d%8d\nRight Tuple %8d%8d\n", lt[i].id, lt[i].val, rt[j].id, rt[j].val);
      //printf("New Tuple %8d%8d%8d%8d\n", p[n].rid, p[n].lid, p[n].rval, p[n].lval);
    }
  }    
    
}

}
