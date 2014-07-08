#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {

__device__
uint search(BUCKET *b,int num,uint right){
  int m,l,r;
  l=0;
  r=right-1;
  do{
    m=(l+r)/2;
    if(num < b[m].val)r=m-1;else l=m+1;
  }while(l<=r&&num!=b[m].val);

  return m;

}

__global__
void count(
          TUPLE *lt,
          uint *count,
          BUCKET *bucket,
          int right,
          int left
          ) 

{
    


  //i,jの方向を間違えないように
  /*
   *x軸が縦の方向、y軸が横の方向だよ。
   *だから、xがleft、yがrightに対応しているよ
   */

  int start = blockIdx.y*MAX_IDX;

  __shared__ BUCKET index[MAX_IDX];
  for(uint i=0 ; i<MAX_IDX&&start+i<right ; i+=blockDim.x){
    index[i] = bucket[start+i];
  }

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  for(uint i=0; i<LOOP ; i++){
    if(x*LOOP+i < left){
      int idx = lt[x*LOOP+i].val;
      uint temp = 0;
      uint bidx = search(index,idx,MAX_IDX);
      uint seq = bidx;
      while(index[seq].val == idx){
        temp++;
        if(seq == 0) break;
        seq--;
      }
      seq = bidx+1;
      while(index[seq].val == idx){
        temp++;
        if(seq == right-1) break;
        seq++;
      }
      count[x] = temp;
      if(x*LOOP+i == left-1){
        count[x+1] = 0;
      }
    }
  }
}


__global__ void join(
          TUPLE *rt,
          TUPLE *lt,
          RESULT *jt,
          uint *count,
          BUCKET *bucket,
          int right,
          int left
          ) 
{


  int start = blockIdx.y*MAX_IDX;

  __shared__ BUCKET index[MAX_IDX];
  for(uint i=0 ; i<MAX_IDX&&start+i<right ; i+=blockDim.x){
    index[i] = bucket[start+i];
  }
  __syncthreads();

  int x = blockIdx.x * blockDim.x + threadIdx.x;

  uint writeloc = count[x];

  for(uint i=0; i<LOOP ; i++){
    if(x*LOOP+i < left){
      int idx = lt[x*LOOP+i].val;
      int lkey = lt[x*LOOP+i].key;
      uint bidx = search(index,idx,MAX_IDX);
      uint seq = bidx;
      uint i = 0;
      while(index[seq].val == idx){
        jt[writeloc + i].rkey = lkey;
        jt[writeloc + i].lkey = rt[index[seq].adr].key;
        jt[writeloc + i].rval = idx;
        jt[writeloc + i].lval = rt[index[seq].adr].val;      
        i++;
        if(seq == 0) break;
        seq--;
      }
      seq = bidx+1;
      while(index[seq].val == idx){
        jt[writeloc + i].rkey = lkey;
        jt[writeloc + i].rval = idx;
        jt[writeloc + i].lkey = rt[index[seq].adr].key;
        jt[writeloc + i].lval = rt[index[seq].adr].val;      
        i++;
        if(seq == right-1) break;
        seq++;
      }
    }
  }

}    

}
