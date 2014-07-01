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

#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/scan.h>
#include "scan_common.h"

int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

    CUdeviceptr *d_Input, *d_Output;
    uint *h_Input, *h_OutputCPU, *h_OutputGPU;
    StopWatchInterface  *hTimer = NULL;
    const uint MIN = 16 * 1024 * 1024;
    const uint MAX = 16 * 1024 * 1024; //max size

    sdkCreateTimer(&hTimer);

    printf("Initializing CUDA-C scan...\n\n");
    initScan();

    int globalFlag = 1;
    size_t szWorkgroup;
    const int iCycles = 5;

    /*
    printf("*** Running GPU scan for short arrays (%d identical iterations)...\n\n", iCycles);

    for (uint arrayLength = MIN_SHORT_ARRAY_SIZE; arrayLength <= MAX_SHORT_ARRAY_SIZE; arrayLength <<= 1)
    {
        printf("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);

        for (int i = 0; i < iCycles; i++)
        {
            szWorkgroup = scanExclusiveShort(d_Output, d_Input, arrayLength);
        }

        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&hTimer);
        double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer) / iCycles;

        printf("Validating the results...\n");
        printf("...reading back GPU results\n");
        checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

        printf(" ...scanExclusiveHost()\n");
        scanExclusiveHost(h_OutputCPU, h_Input, arrayLength);

        // Compare GPU results with CPU results and accumulate error for this test
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

        // Data log
        if (arrayLength == MAX_SHORT_ARRAY_SIZE)
        {
            printf("\n");
            printf("scan-Short, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
                   (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
            printf("\n");
        }
    }


    printf("***Running GPU scan for large arrays (%u identical iterations)...\n\n", iCycles);

    for (uint arrayLength = MIN_LARGE_ARRAY_SIZE; arrayLength <= MAX_LARGE_ARRAY_SIZE; arrayLength <<= 1)
    {
        printf("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);

        for (int i = 0; i < iCycles; i++)
        {
          szWorkgroup = scanExclusiveLarge(d_Output, d_Input, arrayLength);
        }

        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&hTimer);
        double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer) / iCycles;

        printf("Validating the results...\n");
        printf("...reading back GPU results\n");
        checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

        printf("...scanExclusiveHost()\n");
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);

        scanExclusiveHost(h_OutputCPU, h_Input, N);

        sdkStopTimer(&hTimer);
        double CPUtimerValue = 1.0e-3 * sdkGetTimerValue(&hTimer);


        // Compare GPU results with CPU results and accumulate error for this test
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

        // Data log
        if (arrayLength == MAX_LARGE_ARRAY_SIZE)
        {
            printf("\n");
            printf("scan-Large, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
                   (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
            printf("\n");
            printf("CPU Time = %.5f s\n",CPUtimerValue);
        }
    }
    */

    for(uint arrayLength = MIN ; arrayLength <= MAX; arrayLength <<= 1){


      printf("\n\n***********Starting scan for array size %u*************\n",arrayLength);

      printf("Allocating and initializing host arrays...\n");

      int temp = MIN;
      h_Input     = (uint *)malloc(temp * sizeof(uint));
      h_OutputCPU = (uint *)malloc(arrayLength * sizeof(uint));
      h_OutputGPU = (uint *)malloc(arrayLength * sizeof(uint));

      for (uint i = 0; i < temp-1; i++)
        {
          h_Input[i] = 1;//rand();
        }
      h_Input[temp-1] = 0;
      
      printf("Allocating and initializing CUDA arrays...\n");
      checkCudaErrors(cudaMalloc((void **)&d_Input, temp * sizeof(uint)));
      checkCudaErrors(cudaMalloc((void **)&d_Output, arrayLength * sizeof(uint)));
      checkCudaErrors(cudaMemcpy(d_Input, h_Input, temp * sizeof(uint), cudaMemcpyHostToDevice));
      
      
      printf("Running scan for %u elements (1 arrays)...\n", arrayLength);
      checkCudaErrors(cudaDeviceSynchronize());
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
      
      for (int i = 0; i < iCycles; i++)
        {
          szWorkgroup = scanExclusiveLL((uint *)d_Output, (uint *)d_Input, arrayLength);
        }
      
      checkCudaErrors(cudaDeviceSynchronize());
      sdkStopTimer(&hTimer);
      double timerValue = sdkGetTimerValue(&hTimer) / iCycles;
      
      //printf("Validating the results...\n");
      //printf("...reading back GPU results\n");
      checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, arrayLength * sizeof(uint), cudaMemcpyDeviceToHost));
      
      //printf("...scanExclusiveHost()\n");
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);

      for(uint i=0 ;i<iCycles;i++){
        scanExclusiveHost(h_OutputCPU, h_Input, arrayLength);
      }
      sdkStopTimer(&hTimer);
      double CPUtimerValue = sdkGetTimerValue(&hTimer)/iCycles;
      
      
      // Compare GPU results with CPU results and accumulate error for this test
      //printf(" ...comparing the results\n");
      int localFlag = 1;
      
      for (uint i = 0; i < temp; i++)
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
      
      // Data log
      /*
      printf("\n");
      printf("scan-Large, Throughput = %.4f MElements/s, Time = %.3f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
             (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
      printf("\n");
      printf("CPU Time = %.5f s\n",CPUtimerValue);
      printf("GPU = %d\tCPU = %d\n",h_OutputGPU[arrayLength-1],h_OutputCPU[arrayLength-1]);
      */

      printf("GPU = %.3f ms\tCPU = %.3f ms\n",timerValue,CPUtimerValue);


      checkCudaErrors(cudaFree(d_Output));
      checkCudaErrors(cudaFree(d_Input));
      free(h_Input);
      free(h_OutputGPU);
      free(h_OutputCPU);


    }
    
    printf("Shutting down...\n");
    closeScan();


    sdkDeleteTimer(&hTimer);

    cudaDeviceReset();
    // pass or fail (cumulative... all tests in the loop)
    exit(globalFlag ? EXIT_SUCCESS : EXIT_FAILURE);
}
