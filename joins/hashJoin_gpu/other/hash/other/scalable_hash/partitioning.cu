/*
count the number of match tuple in each partition and each thread

*/

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"


extern "C" {

inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1)
    {
        __syncthreads();
        uint t = s_Data[pos] + s_Data[pos - 32 * offset];
        __syncthreads();
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size)
{
    return scan1Inclusive(idata, s_Data, size) - idata;
}


__global__
void count_partitioning(
          TUPLE *t,
          uint *blockCount,
          uint *localScan,
          uint loop,
          uint rows
          ) 

{

  __shared__ uint local[2*PARTITION];
  for(uint i = threadIdx.x; i<2*PARTITION ; i+=blockDim.x){
    local[i] = 0;
  }

  __syncthreads();

  uint firstpos = ONE_BL_NUM*blockIdx.x + threadIdx.x*ONE_BL_NUM/blockDim.x;
  
  for(uint i = firstpos; i<firstpos + ONE_BL_NUM/blockDim.x && i<rows; i++){    
    int temp = t[i].val;
    temp >>= RADIX*loop;
    uint idx = temp%PARTITION;
    atomicAdd(&(local[idx+PARTITION]),1);

  }

  __syncthreads();

  for(uint i=threadIdx.x ; i<PARTITION ; i+=blockDim.x){
    blockCount[gridDim.x*i + blockIdx.x] = local[i+PARTITION];
  }

  __syncthreads();
  
  //scan1Exclusive(local[threadIdx.x+PARTITION],local,PARTITION);
  /*
  for(uint i=threadIdx.x*ONE_BL_NUM/blockDim.x+1; i<(threadIdx.x+1)*ONE_BL_NUM/blockDim.x; i++){
    local[i]=local[i-1]+local[PARTITION+i-1];
  }
  scan1Exclusive(local[threadIdx.x+PARTITION],local,blockDim.x);
  */
  if(threadIdx.x==0){
    local[0]=0;
    for(uint i=1 ; i<PARTITION ; i++){
      local[i]=local[i-1]+local[PARTITION+i-1];
    }
  }
  __syncthreads();

  for(uint i=threadIdx.x ; i<PARTITION ; i+=blockDim.x){
    localScan[gridDim.x*i + blockIdx.x] = local[i];
  }


}

__global__
void partitioning1(
          TUPLE *t,
          TUPLE *pt,
          uint *localScan,
          uint loop,
          uint rows
          ) 

{

  __shared__ uint local[PARTITION];
  for(uint i = threadIdx.x; i<PARTITION ; i+=blockDim.x){
    local[i] = localScan[gridDim.x*i + blockIdx.x];
  }

  __syncthreads();

  uint firstpos = ONE_BL_NUM*blockIdx.x + threadIdx.x;

  for(uint i = firstpos; i < firstpos+ONE_BL_NUM && i<rows; i+=blockDim.x){
    TUPLE temp = t[i];
    int tval = temp.val;
    tval >>= RADIX*loop;
    uint idx = tval%PARTITION;

    uint outlocal = atomicAdd(&(local[idx]),1);
    /*
    uint x;
    for(uint j=0 ; j<32 ; j++){
      x = __shfl(idx,j);
      if(x==idx){
        
      }
    }
    */
    /*
    for(uint j = 0 ; j<32 ; j++){
      if(threadIdx.x==j){
        outlocal = local[idx]++;    
        //outlocal = atomicAdd(&(local[idx]),1);
      }
    }
    */
    pt[ONE_BL_NUM*blockIdx.x + outlocal] = temp;
  }

}


__global__
void partitioning2(
          TUPLE *t,
          TUPLE *pt,
          uint *blockCount,
          uint loop,
          uint rows
          ) 

{

  __shared__ uint local[PARTITION];
  for(uint i = threadIdx.x; i<PARTITION ; i+=blockDim.x){
    local[i] = blockCount[gridDim.x*i + blockIdx.x];
  }
  __syncthreads();

  uint firstpos = ONE_BL_NUM*blockIdx.x + threadIdx.x;

  for(uint i = firstpos; i<ONE_BL_NUM+firstpos && i<rows; i+=blockDim.x){
    TUPLE temp = pt[i];
    int tval = temp.val;
    tval >>= RADIX*loop;
    uint idx = tval%PARTITION;
    uint outlocal = atomicAdd(&(local[idx]),1);
    /*
    for(uint j = 0 ; j<32 ; j++){
      if(threadIdx.x==j){
        outlocal = local[idx]++;
        //outlocal = atomicAdd(&(local[idx]),1);
      }
    }
    */
    t[outlocal] = temp;

  }  

}

  /*
__global__
void partitioningF(
          TUPLE *t,
          TUPLE *pt,
          uint *blockCount,
          uint *startPos,
          uint loop,
          uint rows
          ) 

{

  __shared__ uint local[PARTITION];
  for(uint i = threadIdx.x; i<PARTITION ; i+=blockDim.x){
    local[i] = blockCount[gridDim.x*i + blockIdx.x];
  }
  __syncthreads();

  uint firstpos = ONE_BL_NUM*blockIdx.x + threadIdx.x*ONE_BL_NUM/blockDim.x;

  for(uint i = firstpos; i<ONE_BL_NUM/blockDim.x+firstpos && i<rows; i++){
    TUPLE temp = pt[i];
    int tval = temp.val;
    int part = 1;
    for(uint j=0 ; j<loop ;j++){
      part *= PARTITION;
    }
    int pidx = temp.val%part;
    startPos[pidx]++;
    tval >>= RADIX*loop;
    uint idx = tval%PARTITION;
    t[local[idx]] = temp;
    atomicAdd(&(local[idx]),1);
  }  

}
  */

}
