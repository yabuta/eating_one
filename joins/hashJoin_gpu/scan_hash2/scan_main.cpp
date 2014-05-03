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

    /*
    uint *d_Input, *d_Output;
    uint *h_Input, *h_OutputCPU, *h_OutputGPU;
    const uint N = 2 * 64 * 1048576 ;//13 * 1048576 / 2;
    printf("Allocating and initializing host arrays...\n");
    h_Input     = (uint *)malloc(N * sizeof(uint));
    h_OutputCPU = (uint *)malloc(N * sizeof(uint));
    h_OutputGPU = (uint *)malloc(N * sizeof(uint));
    srand(2009);

    for (uint i = 0; i < N; i++)
    {
      h_Input[i] = 1;//rand();
    }

    printf("Allocating and initializing CUDA arrays...\n");
    checkCudaErrors(cudaMalloc((void **)&d_Input, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, N * sizeof(uint), cudaMemcpyHostToDevice));
    */

    //printf("Initializing CUDA-C scan...\n\n");
    initScan();

    int res = 1;
    size_t szWorkgroup;
    //const int iCycles = 5;


    if(arrayLength <= MAX_SHORT_ARRAY_SIZE && arrayLength > MIN_SHORT_ARRAY_SIZE)
      {    
        //printf("*** Running GPU scan for short arrays (%d identical iterations)...\n\n", 1);

        for(uint i = 4; i<=MAX_SHORT_ARRAY_SIZE ; i<<1){
          if(arrayLength <= i){
            N = i;
          }
        }
        checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));

    
        //printf("Running scan for %u elements (%u arrays)...\n", N , 1);
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);

        szWorkgroup = scanExclusiveShort((uint *)d_Output, (uint *)d_Input, 1, N);

        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&hTimer);
        double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer);

        //printf("Validating the results...\n");
        //printf("...reading back GPU results\n");
        //checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

        /*
        printf(" ...scanExclusiveHost()\n");
        scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);
        */
        // Compare GPU results with CPU results and accumulate error for this test
        /*
        printf(" ...comparing the results\n");
        int localFlag = 1;

        for (uint i = 0; i < N; i++)
        {
            if (h_OutputCPU[i] != h_OutputGPU[i])
            {
                localFlag = 0;
                break;
            }
        }
        */

        // Log message on individual test result, then accumulate to global flag
        //printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
        //globalFlag = globalFlag && localFlag;

        // Data log
        /*
        printf("\n");
        printf("scan-Short, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
               (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
        printf("\n");
        */
    }else if(arrayLength <= MAX_LARGE_ARRAY_SIZE)
    {
      //printf("***Running GPU scan for large arrays (%u identical iterations)...\n\n", 1);

      for(uint i = MIN_LARGE_ARRAY_SIZE; i<=MAX_LARGE_ARRAY_SIZE ; i<<1){
        if(arrayLength <= i){
          N = i;
        }
      }
      checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));      
      
      //printf("Running scan for %u elements (%u arrays)...\n", N, 1);
      checkCudaErrors(cudaDeviceSynchronize());
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
      
      szWorkgroup = scanExclusiveLarge((uint *)d_Output, (uint *)d_Input, 1, N);
      
      checkCudaErrors(cudaDeviceSynchronize());
      sdkStopTimer(&hTimer);
      double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer);

      /*
      printf("Validating the results...\n");
      printf("...reading back GPU results\n");
      checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

      
      printf("...scanExclusiveHost()\n");
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
      scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);
      
      sdkStopTimer(&hTimer);
      double CPUtimerValue = 1.0e-3 * sdkGetTimerValue(&hTimer);
      */

      
      // Compare GPU results with CPU results and accumulate error for this test
      /*
      printf(" ...comparing the results\n");
      int localFlag = 1;
      
      for (uint i = 0; i < N; i++)
        {
          if (h_OutputCPU[i] != h_OutputGPU[i])
            {
              localFlag = 0;
              break;
            }
        }
      
      // Log message on individual test result, then accumulate to global flag
      printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
      globalFlag = globalFlag && localFlag;
      */
      // Data log
      /*
      printf("\n");
      printf("scan-Large, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
             (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
      printf("\n");
      */
    }else if(arrayLength <= MAX_LL_SIZE)
      {
        //printf("***Running GPU scan for LL arrays (%u identical iterations)...\n\n", arrayLength);

        for(uint i = MIN_LL_SIZE; i<=MAX_LL_SIZE ; i<<=1){
          if(arrayLength <= i){
            N = i;
          }
        }
        N = MAX_LARGE_ARRAY_SIZE * ( arrayLength/MAX_LARGE_ARRAY_SIZE + 1 );
        checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));      
        
        //printf("Running scan for %u elements (%u arrays)...\n", arrayLength, 1);
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);
        szWorkgroup = scanExclusiveLL((uint *)d_Output, (uint *)d_Input, 1, N);
        
        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&hTimer);
        double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer);

        /*
        printf("Validating the results...\n");
        printf("...reading back GPU results\n");
        checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

        printf("...scanExclusiveHost()\n");
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);

        scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);

        sdkStopTimer(&hTimer);
        double CPUtimerValue = 1.0e-3 * sdkGetTimerValue(&hTimer);
        */

        // Compare GPU results with CPU results and accumulate error for this test
        /*
        printf(" ...comparing the results\n");
        int localFlag = 1;

        for (uint i = 0; i < N; i++)
        {
            if (h_OutputCPU[i] != h_OutputGPU[i])
            {
                localFlag = 0;
                break;
            }
        }

        // Log message on individual test result, then accumulate to global flag
        printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
        globalFlag = globalFlag && localFlag;
        */
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
    //checkCudaErrors(cudaFree(d_Output));
    sdkDeleteTimer(&hTimer);

    cudaDeviceReset();
    // pass or fail (cumulative... all tests in the loop)
    if(d_Output){
      cuMemFree(d_Input);
    }
    return d_Output;
}
