#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {

__device__
int search(BUCKET *b,int num,uint left,uint right){
  int m,l,r;
  l=left;
  r=right;
  do{
    m=(l+r)/2;
    if(num < b[m].val)r=m-1;else l=m+1;
    if(num==b[m].val){
      return m;
    }
  }while(l<=r);

  return -1;

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
    

  int x = blockIdx.x;
  __shared__ TUPLE t;
  t = lt[x];
  __syncthreads();

  int oneth = right/blockDim.x;
  oneth = right%blockDim.x==0 ? oneth:(oneth+1);

  uint start = threadIdx.x*oneth;
  uint end = (threadIdx.x+1)*oneth;
  end = end<right ? end:right-1;

  int idx = t.val;
  int bidx = search(bucket,idx,start,end);
  if(bidx >= 0){
    uint temp = 0;
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
  if(x == left-1&&threadIdx.x==0){
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


  int x = blockIdx.x;
  __shared__ TUPLE t; 
  t = lt[x];
  __syncthreads();

  int oneth = right/blockDim.x;
  oneth = right%blockDim.x==0 ? oneth:(oneth+1);

  uint start = threadIdx.x*oneth;
  uint end = (threadIdx.x+1)*oneth;
  end = end<right ? end:right-1;

  uint writeloc = 0;
  if(x > 0){
    writeloc = count[x-1];
  }

  int idx = t.val;
  int bidx = search(bucket,idx,start,end);
  if(bidx >= 0){
    uint lkey = t.key;
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
