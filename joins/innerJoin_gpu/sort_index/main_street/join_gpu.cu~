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

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x < left){
    int idx = lt[x].val;
    uint temp = 0;
    uint bidx = search(bucket,idx,right);
    uint seq = bidx;
    while(bucket[seq].val == idx){
      temp++;
      if(seq == 0) break;
      seq--;
    }
    seq = bidx+1;
    while(bucket[seq].val == idx){
      temp++;
      if(seq == right-1) break;
      seq++;
    }
    count[x] = temp;
  }

  if(x == left-1){
    count[x+1] = 0;
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


  int x = blockIdx.x * blockDim.x + threadIdx.x;


  uint writeloc = count[x];

  if(x < left){
    int idx = lt[x].val;
    uint lkey = lt[x].key;
    uint bidx = search(bucket,idx,right);
    uint seq = bidx;
    uint i = 0;
    while(bucket[seq].val == idx){
      jt[writeloc + i].rkey = lkey;
      jt[writeloc + i].lkey = rt[bucket[seq].adr].key;
      jt[writeloc + i].rval = idx;
      jt[writeloc + i].lval = rt[bucket[seq].adr].val;      
      i++;
      if(seq == 0) break;
      seq--;
    }
    seq = bidx+1;
    while(bucket[seq].val == idx){
      jt[writeloc + i].rkey = lkey;
      jt[writeloc + i].rval = idx;
      jt[writeloc + i].lkey = rt[bucket[seq].adr].key;
      jt[writeloc + i].lval = rt[bucket[seq].adr].val;      
      i++;
      if(seq == right-1) break;
      seq++;
    }
  }

}    

}
