#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {

__device__
uint search(BUCKET *b,int num,uint right){
  uint m,l,r;
  l=1;
  r=right;
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


  int y = blockIdx.y * blockDim.y + threadIdx.y;


  uint writeloc = 0;
  if(y > 0){
    writeloc = count[y-1];
  }

  if(y < left){
    int idx = lt[y].val % NB_BKT_ENT;
    uint bidx = search(bucket,idx,right);
    uint x = bidx;
    uint i = 0;
    while(bucket[x].val == idx){
      jt[writeloc + i].rkey = lt[y].key;
      jt[writeloc + i].rval = lt[y].val;
      jt[writeloc + i].lkey = rt[bucket[x].adr].key;
      jt[writeloc + i].lval = rt[bucket[x].adr].val;      
      i++;
      x--;
    }
    x = bidx;
    while(bucket[x].val == idx){
      jt[writeloc + i].rkey = lt[y].key;
      jt[writeloc + i].rval = lt[y].val;
      jt[writeloc + i].lkey = rt[bucket[x].adr].key;
      jt[writeloc + i].lval = rt[bucket[x].adr].val;      
      i++;
      x++;
    }
  }

}    

}
