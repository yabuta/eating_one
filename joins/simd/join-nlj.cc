#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>//AVX: -mavx

typedef struct _TPL {
  int key;
  int val;
} TPL;

#define SZ_R (1024*128)
#define SZ_S (1024*128)

TPL R[SZ_R];
TPL S[SZ_S];

int *BufSimdR;
int *BufSimdS;
int *BufSimdT;
const int SzBufSimd = 1024;

void 
vec_add(const size_t N, int *t, const int *r, const int *s)
{
  static const size_t single_size = 8; 
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
  static const size_t single_size = 8; // 8 instructions
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
  int i;

  for (i = 0; i < SZ_R; i++) {
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
printResult(TPL t)
{
  printf("%d\n", t.key);
}

int 
main(void)
{
  init();

  for (int i = 0; i < SZ_R; i ++) {
    for(int j = 0; j < SZ_S; j++) {
      if (R[i].val == S[j].val) printResult(R[i]);
    }
  }

  cleanup();

  return 0;
}
