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


  uint writeloc = 0;
  if(x > 0){
    writeloc = count[x-1];
  }

  if(x < left){
    int idx = lt[x].val;
    uint lkey = lt[x].key;
    uint bidx = search(bucket,idx,right);
    uint seq = bidx;
    while(bucket[seq].val == idx){
      jt[writeloc].rkey = lkey;
      jt[writeloc].rval = idx;
      jt[writeloc].lkey = rt[bucket[seq].adr].key;
      jt[writeloc].lval = rt[bucket[seq].adr].val;      
      writeloc++;
      if(seq == 0) break;
      seq--;
    }
    seq = bidx+1;
    while(bucket[seq].val == idx){
      jt[writeloc].rkey = lkey;
      jt[writeloc].rval = idx;
      jt[writeloc].lkey = rt[bucket[seq].adr].key;
      jt[writeloc].lval = rt[bucket[seq].adr].val;      
      writeloc++;
      if(seq == right-1) break;
      seq++;
    }
  }

}    

}
