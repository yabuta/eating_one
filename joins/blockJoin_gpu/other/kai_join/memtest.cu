#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "tuple.h"

extern "C" {
__global__

void test(char *string){

  printf("string=%s\n",string);

}


}


