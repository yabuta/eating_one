/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


/*
yabuta's after comment

this file arrange interface of scan kernel. 

presum calls a suit kernel per size.

part_presum calculate difference of discrete element of array.


 */

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
//#include <cuda_runtime.h>
#include "scan_common.h"

static uint iDivUp(uint dividend, uint divisor)
{
    return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

uint presum(CUdeviceptr *d_Input, uint arrayLength)
{
  //printf("Starting...\n\n");

  //StopWatchInterface  *hTimer = NULL;
    //sdkCreateTimer(&hTimer);

    uint N = 0;
    CUdeviceptr d_Output;

    //printf("Initializing CUDA-C scan...\n\n");
    initScan();
    //size_t szWorkgroup;

    if(arrayLength <= MAX_SHORT_ARRAY_SIZE && arrayLength > MIN_SHORT_ARRAY_SIZE)
      {    
        for(uint i = 4; i<=MAX_SHORT_ARRAY_SIZE ; i<<=1){
          if(arrayLength <= i){
            N = i;
            break;
          }
        }
        checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));

        checkCudaErrors(cudaDeviceSynchronize());

        scanExclusiveShort((uint *)d_Output, (uint *)(*d_Input), N);
        //szWorkgroup = scanExclusiveShort((uint *)d_Output, (uint *)d_Input, 1, N);

        checkCudaErrors(cudaDeviceSynchronize());

        // Data log
        /*
        printf("\n");
        printf("scan-Short, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
               (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
        printf("\n");
        */
    }else if(arrayLength <= MAX_LARGE_ARRAY_SIZE)
    {

      /*
      for(uint i = MIN_LARGE_ARRAY_SIZE; i<=MAX_LARGE_ARRAY_SIZE ; i<<=1){
        if(arrayLength <= i){
          N = i;
          break;
        }
      }
      */

      N = MAX_SHORT_ARRAY_SIZE * iDivUp(arrayLength,MAX_SHORT_ARRAY_SIZE);

      checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));      
      
      checkCudaErrors(cudaDeviceSynchronize());

      scanExclusiveLarge((uint *)d_Output, (uint *)(*d_Input), N);
      //szWorkgroup = scanExclusiveLarge((uint *)d_Output, (uint *)d_Input, 1, N);
      
      checkCudaErrors(cudaDeviceSynchronize());

      /*
      printf("\n");
      printf("scan-Large, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
             (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
      printf("\n");
      */
    }else if(arrayLength <= MAX_LL_SIZE)
      {


        N = MAX_LARGE_ARRAY_SIZE * iDivUp(arrayLength,MAX_LARGE_ARRAY_SIZE);

        checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));      
        
        checkCudaErrors(cudaDeviceSynchronize());

        scanExclusiveLL((uint *)d_Output, (uint *)(*d_Input), N);
        //szWorkgroup = scanExclusiveLL((uint *)d_Output, (uint *)d_Input, 1, N);
        
        checkCudaErrors(cudaDeviceSynchronize());

        // Data log
        /*
        printf("\n");
        printf("scan-Large, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
               (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
        printf("\n");
        */
      }else{
      cuMemFree(d_Output);
      closeScan();

      return FALSE;      
    }

    closeScan();

    cuMemFree(*d_Input);
    *d_Input = d_Output;

    //return *d_Input;
    return SUCCESS;
}


/**
part_presum
  first: CUdeviceptr
  second: the size of a space
  third: the scaned array size
  forth: table size

 **/

CUdeviceptr diff_part(CUdeviceptr d_Input , uint tnum , uint arrayLength, uint size){

  
  CUdeviceptr d_Output;
  
  checkCudaErrors(cudaMalloc((void **)&d_Output, arrayLength * sizeof(uint)));            
  checkCudaErrors(cudaDeviceSynchronize());
  
  diff_Part((uint *)d_Output, (uint *)d_Input, tnum, arrayLength,size);
  //szWorkgroup = scanExclusiveLarge((uint *)d_Output, (uint *)d_Input, pnum, N);
  checkCudaErrors(cudaDeviceSynchronize());
  
  //cudaDeviceReset();
  // pass or fail (cumulative... all tests in the loop)  

  return d_Output;

}

CUdeviceptr transport(CUdeviceptr d_Input , uint tnum , uint arrayLength, uint size){

  CUdeviceptr d_Output;
  
  checkCudaErrors(cudaMalloc((void **)&d_Output, (arrayLength+1) * sizeof(uint)));
  checkCudaErrors(cudaDeviceSynchronize());

  transport_gpu((uint *)d_Output, (uint *)d_Input, tnum, arrayLength,size);
  //szWorkgroup = scanExclusiveLarge((uint *)d_Output, (uint *)d_Input, pnum, N);
  checkCudaErrors(cudaDeviceSynchronize());

  // pass or fail (cumulative... all tests in the loop)

  return d_Output;


}

uint getValue(CUdeviceptr d_Input , uint loc , uint *res){

  CUdeviceptr d_Output;
  
  checkCudaErrors(cudaMalloc((void **)&d_Output, sizeof(int)));
  checkCudaErrors(cudaDeviceSynchronize());

  getValue_gpu((uint *)d_Output, (uint *)d_Input, loc);
  checkCudaErrors(cudaDeviceSynchronize());

  if(cuMemcpyDtoH(res,d_Output,sizeof(uint)) != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(d_Output) error.\n");
    exit(1);
  }

  cuMemFree(d_Output);

  return SUCCESS;


}
