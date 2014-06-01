// Debug MACRO
#define C(val)  do {fprintf(stderr, "%16s %4d %16s %16s: %c\n", __FILE__, __LINE__, __func__, #val, val); fflush(stderr);} while (0)
#define D(val)  do {fprintf(stderr, "%16s %4d %16s %16s: %d\n", __FILE__, __LINE__, __func__, #val, val); fflush(stderr);} while (0)
#define P(val)  do {fprintf(stderr, "%16s %4d %16s %16s: %p\n", __FILE__, __LINE__, __func__, #val, val); fflush(stderr);} while (0)
#define L(val)  do {fprintf(stderr, "%16s %4d %16s %16s: %ld\n", __FILE__, __LINE__, __func__, #val, val); fflush(stderr);} while (0)
#define LL(val) do {fprintf(stderr, "%16s %4d %16s %16s: %lld\n", __FILE__, __LINE__, __func__, #val, val); fflush(stderr);} while (0)
#define S(val)  do {fprintf(stderr, "%16s %4d %16s %16s: %s\n", __FILE__, __LINE__, __func__, #val, val); fflush(stderr);} while (0)
#define F(val)  do {fprintf(stderr, "%16s %4d %16s %16s: %f\n", __FILE__, __LINE__, __func__, #val, val); fflush(stderr);} while (0)
#define N       do {fprintf(stderr, "%16s %4d %16s\n",  __FILE__, __LINE__, __func__); fflush(stderr);} while (0)
#define ERR     do {perror("ERROR"); N; exit(1);} while (0)

