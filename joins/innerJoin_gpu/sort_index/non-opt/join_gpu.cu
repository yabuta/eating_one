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
    
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x < left){
    uint bidx = search(bucket,lt[x].val,right);
    uint seq = bidx;
    count[x]=0;
    while(bucket[seq].val == lt[x].val){
      count[x]++;
      if(seq == 0) break;
      seq--;
    }
    seq = bidx+1;
    while(bucket[seq].val == lt[x].val){
      count[x]++;
      if(seq == right-1) break;
      seq++;
    }
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


  if(x < left){
    uint bidx = search(bucket,lt[x].val,right);
    uint seq = bidx;
    while(bucket[seq].val == lt[x].val){
      jt[count[x]].rkey = lt[x].key;
      jt[count[x]].lkey = rt[bucket[seq].adr].key;
      jt[count[x]].rval = lt[x].val;
      jt[count[x]].lval = rt[bucket[seq].adr].val;      
      count[x]++;
      if(seq == 0) break;
      seq--;
    }
    seq = bidx+1;
    while(bucket[seq].val == lt[x].val){
      jt[count[x]].rkey = lt[x].key;
      jt[count[x]].rval = lt[x].val;
      jt[count[x]].lkey = rt[bucket[seq].adr].key;
      jt[count[x]].lval = rt[bucket[seq].adr].val;      
      count[x]++;
      if(seq == right-1) break;
      seq++;
    }
  }

}    

}
