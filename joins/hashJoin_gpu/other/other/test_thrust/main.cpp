#include <stdio.h>
#include <thrust/scan.h>

#define N 1024*1024

int main(){


  int *r = (int *)calloc(N,sizeof(int));
  //int *l = (int *)calloc(1024*1024,sizeof(int));

  for(unsigned int i=0; i<N ;i++){
    r[i] = 1;
  }

  printf("starting scan...\n");

  thrust::exclusive_scan(r,r+N,r);

  printf("...finish\n");
  printf("r[N] = %d\n",r[N-1]);

  return 0;

}
