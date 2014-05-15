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
#include <cuda_runtime.h>
#include "scan_common.h"

CUdeviceptr presum(CUdeviceptr d_Input, uint arrayLength)
{
  //printf("Starting...\n\n");

    StopWatchInterface  *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    uint N = 0;
    CUdeviceptr d_Output;

    //printf("Initializing CUDA-C scan...\n\n");
    initScan();
    //size_t szWorkgroup;

    if(arrayLength <= MAX_SHORT_ARRAY_SIZE && arrayLength > MIN_SHORT_ARRAY_SIZE)
      {    
        for(uint i = 4; i<=MAX_SHORT_ARRAY_SIZE ; i<<1){
          if(arrayLength <= i){
            N = i;
          }
        }
        checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));

        checkCudaErrors(cudaDeviceSynchronize());

        scanExclusiveShort((uint *)d_Output, (uint *)d_Input, 1, N);
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
      for(uint i = MIN_LARGE_ARRAY_SIZE; i<=MAX_LARGE_ARRAY_SIZE ; i<<1){
        if(arrayLength <= i){
          N = i;
        }
      }
      checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));      
      
      checkCudaErrors(cudaDeviceSynchronize());
      
      scanExclusiveLarge((uint *)d_Output, (uint *)d_Input, 1, N);
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
        /*
        for(uint i = MIN_LL_SIZE; i<=MAX_LL_SIZE ; i<<=1){
          if(arrayLength <= i){
            N = i;
          }
        }
        */

        N = MAX_LARGE_ARRAY_SIZE * ( arrayLength/MAX_LARGE_ARRAY_SIZE + 1 );
        checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));      
        
        checkCudaErrors(cudaDeviceSynchronize());

        scanExclusiveLL((uint *)d_Output, (uint *)d_Input, 1, N);
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
      d_Output = NULL;
    }

    //printf("Shutting down...\n");
    closeScan();
    sdkDeleteTimer(&hTimer);

    cudaDeviceReset();
    if(d_Output){
      cuMemFree(d_Input);
    }
    return d_Output;
}


/**
part_presum
  first: CUdeviceptr
  second: the size of a space
  third: the scaned array size

 **/

CUdeviceptr diff_part(CUdeviceptr d_Input , uint pnum , uint arrayLength){

  
  CUdeviceptr d_Output;
  
  checkCudaErrors(cudaMalloc((void **)&d_Output, arrayLength * sizeof(uint)));            
  checkCudaErrors(cudaDeviceSynchronize());
  
  diff_Part((uint *)d_Output, (uint *)d_Input, pnum, arrayLength);
  //szWorkgroup = scanExclusiveLarge((uint *)d_Output, (uint *)d_Input, pnum, N);
  checkCudaErrors(cudaDeviceSynchronize());
  
  //cudaDeviceReset();
  // pass or fail (cumulative... all tests in the loop)
  if(d_Output){
    cuMemFree(d_Input);
  }
  return d_Output;

}
