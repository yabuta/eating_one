#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>//AVX: -mavx

void 
vec_add(const size_t N, int *result, const int *r, const int *s)
{
  static const size_t single_size = 8; 
  const size_t end = N / single_size; 

  __m256 *vz = (__m256 *)result;
  __m256 *vx = (__m256 *)r;
  __m256 *vy = (__m256 *)s;
  
  for(size_t i=0; i<end; ++i) {
    vz[i] = _mm256_add_ps(vx[i], vy[i]);
  }
}

void 
vec_cmp(const size_t N, int *result, const int *r, const int *s)
{
  static const size_t single_size = 8; 
  const size_t end = N / single_size; 

  __m256 *vz = (__m256 *)result;
  __m256 *vx = (__m256 *)r;
  __m256 *vy = (__m256 *)s;
  
  for(size_t i=0; i<end; ++i) {
    vz[i] = _mm256_cmp_ps(vx[i], vy[i], _CMP_EQ_OQ);
  }
}

int main(void)
{
  const size_t N = 1024;
  int *r, *s, *result;
  
  r = (int *)_mm_malloc(sizeof(int) * N, 32);
  s = (int *)_mm_malloc(sizeof(int) * N, 32);
  result = (int *)_mm_malloc(sizeof(int) * N, 32);

  for(size_t i=0; i<N; ++i) r[i] = i;
  for(size_t i=0; i<N; ++i) s[i] = i+1;
  for(size_t i=0; i<N; ++i) result[i] = 0;

  vec_cmp(N, result, r, s);
  
  for(size_t i=0; i<N; ++i) printf("%d\n", result[i]);

  _mm_free(r);
  _mm_free(s);
  _mm_free(result);

  return 0;
}
