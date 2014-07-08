#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>
#include "tuple.h"

extern "C" {

__device__
uint search(BUCKET *b,int *upper,int num,uint right){
  int m,l,r,dep,wid,pos,val,temp;
  l=0;
  r=right-1;
  dep = 1;
  wid = 1;
  do{
    m=(l+r)/2;
    if(dep < DEPTH){
      temp=1;
      for(int i=0;i<dep-1;i++){
        temp *=2; 
      }
      pos = temp-1+wid-1;
      wid = (wid-1)*2;
      val = upper[pos];
      if(num < val){
        r=m-1;
        wid+=1;
      }else{
        l=m+1;
        wid+=2;
      }
      dep++;
    }else{
      val = b[m].val;
      if(num < val)r=m-1;else l=m+1;
    }
  }while(l<=r&&num!=val);

  return m;

}

__global__
void count(
          TUPLE *lt,
          uint *count,
          BUCKET *bucket,
          int *up,
          int right,
          int left
          )
{
    

  int x = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int upper[SHARED_SIZE];
  for(uint i=threadIdx.x ; i<SHARED_SIZE ; i+=blockDim.x){
    upper[i] = up[i];
  }
  __syncthreads();

  
  if(x < left){
    int idx = lt[x].val;
    uint temp = 0;
    uint bidx = search(bucket,upper,idx,right);
    uint seq = bidx;
    while(bucket[seq].val == idx){
      temp++;
      if(seq == 0) break;
      seq--;
    }
    seq = bidx+1;
    if(seq < right){
      while(bucket[seq].val == idx){
        temp++;
        if(seq == right-1) break;
        seq++;
      }
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
          int *up,
          int right,
          int left
          ) 
{


  int x = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int upper[SHARED_SIZE];
  for(uint i=threadIdx.x ; i<SHARED_SIZE ; i+=blockDim.x){
    upper[i] = up[i];
  }
  __syncthreads();

  uint writeloc = count[x];

  if(x < left){
    int idx = lt[x].val;
    uint lkey = lt[x].key;
    uint bidx = search(bucket,upper,idx,right);
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
    if(seq < right){
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

}
