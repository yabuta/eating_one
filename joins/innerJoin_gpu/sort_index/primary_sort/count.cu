/*
count the number of tuple matching criteria for join

*/

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {

__device__
uint search(BUCKET *b,int num,uint right){
  uint m,l,r;
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

}

}
