/*
count the number of tuple matching criteria for join

*/

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"


#define NUM_BANKS 16  
#define LOG_NUM_BANKS 4  
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 


extern "C" {


  __global__ void presum(int *sum, int *count, int num){


    __shared__ int temp[2 * SCAN_SIZE_X]; // allocated on validation  
    //extern __shared__ int temp[]; // allocated on invocation  

    for(int i = 0;i<2 * SCAN_SIZE_X;i++){
      temp[i] = 0;

    }


    int n = 0;
    int thid1 = threadIdx.x + SCAN_SIZE_X * blockIdx.x;
    int thid = threadIdx.x;
    if((gridDim.x-1) * SCAN_SIZE_X > thid1){
      n = SCAN_SIZE_X;
    }else{
      n = num - (gridDim.x-1) * SCAN_SIZE_X;
    }

    int pout = 0, pin = 1;  
    // Load input into shared memory.  
    // This is exclusive scan, so shift right by one  
    // and set first element to 0  
    temp[pout*n + thid] = (thid > 0) ? count[thid1-1] : 0;  

    
    /*
    if(thid > 0){
      printf("%d\t%d\n",thid-1,temp[thid - 1]);
    }
    */
    

    //printf("%d\t%d\n",thid,n);

    __syncthreads();  
    for (int offset = 1; offset < n; offset *= 2){  
      pout = 1 - pout; // swap double buffer indices
      pin = 1 - pout;
      if (thid >= offset) temp[pout*n+thid] += temp[pin*n+thid - offset];
      else temp[pout*n+thid] = temp[pin*n+thid];

      //printf("%d\t%d\n",thid,temp[pout*n + thid]);
        
      __syncthreads();
    }

    sum[thid1] = temp[pout*n+thid]; // write output

    //printf("%d\t%d\n",thid1,sum[thid1]);


    __syncthreads();    
    
    __shared__ int part_sum;
    part_sum = 0;

    if(threadIdx.x == SCAN_SIZE_X-1){
      for(int i = 0;i<blockIdx.x;i++){
        part_sum += sum[(SCAN_SIZE_X+1) * blockIdx.x-1];    
        
      }
    }
    
    sum[thid] += part_sum;


  }

  /*
    __global__ void presum(int *count, int *sum, int n){  


    extern __shared__ int temp[];  // allocated on invocation  
    int thid = threadIdx.x;
    int offset = 1; 

    int ai = thid;
    int bi = thid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);  
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi); 
    temp[ai + bankOffsetA] = count[ai + SCAN_SIZE_X*blockIdx.x];
    temp[bi + bankOffsetB] = count[bi + SCAN_SIZE_X*blockIdx.x];


    for (int d = n>>1; d > 0; d >>= 1){                    // build sum in place up the tree  
      __syncthreads();  
      if (thid < d){  
        int ai = offset*(2*thid+1)-1;  
        int bi = offset*(2*thid+2)-1;  
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);
        
        temp[bi] += temp[ai];  
      }
    }
    offset *= 2; 
    if (thid==0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;}  
    
    
    for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan  
      offset >>= 1;  
      __syncthreads();  
      if (thid < d){  
        
        int ai = offset*(2*thid+1)-1;  
        int bi = offset*(2*thid+2)-1;  
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);  
        
        float t = temp[ai];  
        temp[ai] = temp[bi];  
        temp[bi] += t;   
      }  
    }  
    __syncthreads();


    printf("clear\n");    
    printf("%d\n",temp[ai + bankOffsetA]);
    sum[ai + SCAN_SIZE_X*blockIdx.x] = temp[ai + bankOffsetA];  
    sum[bi + SCAN_SIZE_X*blockIdx.x] = temp[bi + bankOffsetB]; 


  }

  */
}

