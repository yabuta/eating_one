#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {
__global__


//main引数をとるかどうか
#ifdef ARG

void join(
          TUPLE *lt,
          TUPLE *rt,
          JOIN_TUPLE *p,
          // int *ltn,
          // int *rtn
          int ltn,
          int rtn
          ) 

{

  int k;
    
  //i,jの方向を間違えないように
  /*
   *x軸が縦の方向、y軸が横の方向だよ。
   *だから、xがleft、yがrightに対応しているよ
   */

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  //  int n = j * (*rtn) + i;
  int n = j * (rtn) + i;

  // if(n==0) {
  //   printf("ltn %d rtn %d\n", ltn, rtn);
  //   return;
  // }
  
  if(&(lt[j])==NULL||&(rt[i])==NULL||&(p[n])==NULL){
    printf("memory error in .cu.\n");
    return;// -1;
  }
 
  //  if((lt[j].val[0]==rt[i].val[0])&&(j<*ltn)&&(i<*rtn)) {
  if((lt[j].val[0]==rt[i].val[0])&&(j<ltn)&&(i<rtn)) {

    //時間計測はできないそうだよ。タイムスタンプをどうするかな
    //gettimeofday(&p->t, NULL);

    for(k=0; k<VAL_NUM; k++) {
      p[n].lval[k] = lt[j].val[k];
      p[n].rval[k] = rt[i].val[k];  
    }

    // lid & rid are just for debug
    p[n].lid = lt[j].id;
    p[n].rid = rt[i].id;

    /*
    if(i==0){
      //printf("Left Tuple %8d%8d\nRight Tuple %8d%8d\n", lt[i].id, lt[i].val, rt[j].id, rt[j].val);
      //printf("New Tuple %8d%8d%8d%8d\n", p[n].rid, p[n].lid, p[n].rval, p[n].lval);
      }*/
  }    
    
}    

#else

void join(TUPLE *lt,TUPLE *rt,JOIN_TUPLE *p) {
    

  int k;

  //i,jの方向を間違えないように
  /*
   *x軸が縦の方向、y軸が横の方向だよ。
   *だから、yがleft、xがrightに対応しているよ
   */

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int n = j * MAX_LEFT + i;

  if(&(lt[i])==NULL||&(rt[j])==NULL||&(p[n])==NULL){
    printf("memory error in .cu.\n");
    return;// -1;
  }else if((lt[i].val==rt[j].val)&&(i<MAX_LEFT)&&(j<MAX_RIGHT)){
      
    //時間計測はできないそうだよ。タイムスタンプをどうするかな
    //gettimeofday(&p->t, NULL);

    for(k=0;k<VAL_NUM;k++){
      p[n].lval = lt[i].val;
      p[n].rval = rt[j].val;  
    }
    // lid & rid are just for debug
    p[n].lid = lt[i].id;
    p[n].rid = rt[j].id;
      
    if(i==0){
      //printf("Left Tuple %8d%8d\nRight Tuple %8d%8d\n", lt[i].id, lt[i].val, rt[j].id, rt[j].val);
      //printf("New Tuple %8d%8d%8d%8d\n", p[n].rid, p[n].lid, p[n].rval, p[n].lval);
    }
  }    

}
#endif
}
