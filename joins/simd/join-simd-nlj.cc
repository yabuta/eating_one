#include <stdio.h>
#include "debug.h"
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>//AVX: -mavx

typedef struct _TPL {
  int key;
  int val;
} TPL;

const int SzBufSimd = 16;

#define SZ_R (1024*128)
#define SZ_S (1024*128)
#define SZ_SIMD (16)

TPL R[SZ_R];
TPL S[SZ_S];

int *BufSimdR;
int *BufSimdS;
int *BufSimdT;


void 
vec_add(const size_t N, int *t, const int *r, const int *s)
{
  static const size_t single_size = SZ_SIMD; 
  const size_t end = N / single_size; 

  __m256 *vz = (__m256 *)t;
  __m256 *vx = (__m256 *)r;
  __m256 *vy = (__m256 *)s;
  
  for(size_t i=0; i<end; ++i) {
    vz[i] = _mm256_add_ps(vx[i], vy[i]);
  }
}

void vec_cmp(const size_t N, int *t, const int *r, const int *s)
{
  static const size_t single_size = SZ_SIMD;
  const size_t end = N / single_size; 

  __m256 *vz = (__m256 *)t;
  __m256 *vx = (__m256 *)r;
  __m256 *vy = (__m256 *)s;
  
  for(size_t i=0; i<end; ++i) {
    vz[i] = _mm256_cmp_ps(vx[i], vy[i], _CMP_EQ_OQ);
  }
}

void
initSimdRST()
{
  BufSimdR = (int *)_mm_malloc(sizeof(int) * SzBufSimd, 32);
  BufSimdS = (int *)_mm_malloc(sizeof(int) * SzBufSimd, 32);
  BufSimdT = (int *)_mm_malloc(sizeof(int) * SzBufSimd, 32);
}

void
initRS()
{
  for (int i = 0; i < SZ_R; i++) {
    R[i].key = i;
    R[i].val = 0;
  }
  for (int i = 0; i < SZ_S; i++) {
    S[i].key = i;
    S[i].val = 1;
  }
}

void
init()
{
  initRS();
  initSimdRST();
}

void
cleanup()
{
  _mm_free(BufSimdR);
  _mm_free(BufSimdS);
  _mm_free(BufSimdT);
}

void
printResult(int BufSimdT[], int radr, int sadr)
{
  // Result 0 means comparison result is true.
  for(int i = 0; i < SzBufSimd / SZ_SIMD; i++) {
    if (BufSimdT[i] != 0) printf("%d\n", BufSimdT[i]);
    /*
      // The following outputs all the results
    printf("%d %d %d %d\n", 
	   R[radr+i].key, R[radr+i].val, 
	   S[sadr+i].key, S[sadr+i].val);
    */
  }
}

int 
main(void)
{
  int i, j, k, l, m;
  init();

  for (i = 0; i < SZ_R; i ++) {
    for (j = 0; j < SzBufSimd; j++) BufSimdR[j] = R[i].val;
    for (k = 0; k < SZ_S; k += SzBufSimd) {
      for (l = 0; l < SzBufSimd / SZ_SIMD; l++) BufSimdT[l] = 0; // init
      for (m = 0; m < SzBufSimd; m++) BufSimdS[m] = S[k + m].val;

      vec_cmp(SzBufSimd, BufSimdT, BufSimdR, BufSimdS);

      printResult(BufSimdT, i, k);
    }
  }
  cleanup();

  return 0;
}
