#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

texture <int2, cudaTextureType1D, cudaReadModeElementType> plt_tex;
texture <int2, cudaTextureType1D, cudaReadModeElementType> prt_tex;

extern "C" {

__global__
void count(
           //TUPLE *lt,
           //TUPLE *rt,
          int *count,
          int *r_p,
          int *radix,
          int *lp,
          int right,
          int left
          )

{

  int x = blockIdx.x*blockDim.x*gridDim.y + blockDim.x*blockIdx.y + threadIdx.x;

  //insert partition left table in shared memory
  __shared__ TUPLE sub_lt[B_ROW_NUM];

  int lp_front = lp[blockIdx.x];
  int lp_back = lp[blockIdx.x+1];
  int2 fetched_val;

  for(int i=lp_front + threadIdx.x,j=threadIdx.x; i<lp_back; i += blockDim.x, j += blockDim.x){
    //for(int i=lp[blockIdx.x] + threadIdx.x,j=threadIdx.x; i<lp[blockIdx.x+1]; i += blockDim.x, j += blockDim.x){
    if(j<B_ROW_NUM){
      fetched_val = tex1Dfetch(plt_tex, i);
      sub_lt[j].key = fetched_val.x;
      sub_lt[j].val = fetched_val.y;

      //sub_lt[j].key = lt[i].key;
      //sub_lt[j].val = lt[i].val;
    }
  }

  __syncthreads();

  /*
  int x_limit = 0;

  if(r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]]%GRID_SIZE_Y == 0){
    x_limit = (r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]])/GRID_SIZE_Y;
  }else{
    x_limit = (r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]])/GRID_SIZE_Y + 1;
  }
  */

  int temp=0;
  //int temp2 = r_p[radix[blockIdx.x]];
  int temp3 = r_p[radix[blockIdx.x]+1];
  int temp4 = lp[blockIdx.x+1] - lp[blockIdx.x];
  int count_x_temp = 0;

  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<temp3 ; k += blockDim.x){
    fetched_val = tex1Dfetch(prt_tex, k);
    temp = fetched_val.y;
    //temp = rt[k].val;
    for(int i=0; i<temp4 ;i++){
      if(sub_lt[i].val == temp){
        //count[x]++;
        count_x_temp++;
      }
    }
  }

  count[x] = count_x_temp;

}


__global__ void join(
                     //TUPLE *lt,
                     //TUPLE *rt,
          RESULT *jt,
          int *count,
          int *r_p,
          int *radix,
          int *lp,
          int right,
          int left
          ) 
{

  //int x = blockIdx.x*blockDim.x + threadIdx.x;
  int x = blockIdx.x*blockDim.x*gridDim.y + blockDim.x*blockIdx.y + threadIdx.x;
  int2 fetched_val;

  __shared__ TUPLE sub_lt[B_ROW_NUM];

  for(int i=lp[blockIdx.x] + threadIdx.x,j=threadIdx.x; i<lp[blockIdx.x+1]; i += blockDim.x, j += blockDim.x){
    //for(int i=lp_front + threadIdx.x,j=threadIdx.x; i<lp_back; i += blockDim.x, j += blockDim.x){
    if(j<B_ROW_NUM){
      fetched_val = tex1Dfetch(plt_tex, i);
      sub_lt[j].key = fetched_val.x;
      sub_lt[j].val = fetched_val.y;
      
      //sub_lt[j].key = lt[i].key;
      //sub_lt[j].val = lt[i].val;
    }
  }


  __syncthreads();

  /*
  int x_limit = 0;

  if(r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]]%GRID_SIZE_Y == 0){
    x_limit = (r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]])/GRID_SIZE_Y;
  }else{
    x_limit = (r_p[radix[blockIdx.x]+1] - r_p[radix[blockIdx.x]])/GRID_SIZE_Y + 1;
  }
  */
  
  TUPLE temp;
  //int temp2 = r_p[radix[blockIdx.x]];
  int temp3 = r_p[radix[blockIdx.x]+1];
  int temp4 = lp[blockIdx.x+1] - lp[blockIdx.x];
  int tcount=count[x];

  for(int k=r_p[radix[blockIdx.x]]+threadIdx.x ; k<temp3 ; k += blockDim.x){
    fetched_val = tex1Dfetch(prt_tex, k);
    temp.key = fetched_val.x;
    temp.val = fetched_val.y;

    //temp.key = rt[k].key;
    //temp.val = rt[k].val;

    for(int i=0; i<temp4 ;i++){
      if(sub_lt[i].val == temp.val){

        jt[tcount].rkey = temp.key;
        jt[tcount].rval = temp.val;
        jt[tcount].lkey = sub_lt[i].key;
        jt[tcount].lval = sub_lt[i].val;

        /*
        temp.key = sub_lt[i].key;
        temp.val = sub_lt[i].val;
        temp.key = sub_lt[i].key;
        temp.val = sub_lt[i].val;
        */
        tcount++;

      }
    }
  }

    
}    

}
