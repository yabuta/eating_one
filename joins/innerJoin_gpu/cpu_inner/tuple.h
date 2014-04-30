/*
#define SZ_PAGE 131072
#define NB_BUFR (SZ_PAGE * 2 / sizeof(TUPLE))
#define NB_BUFS (SZ_PAGE * 16 / sizeof(TUPLE))
*/

#define NUM_VAL 1

#define JT_SIZE 1200000000
#define SELECTIVITY 100000000


typedef struct _TUPLE {
  int key;
  int val;
} TUPLE;

typedef struct _RESULT {
  int rkey;
  int rval;
  int skey;
  int sval;
} RESULT;

typedef struct _BUCKET {
  int adr;
  int val;

} BUCKET;

#define NB_BUCKET 1310720
