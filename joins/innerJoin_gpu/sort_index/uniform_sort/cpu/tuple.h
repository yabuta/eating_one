/*
#define SZ_PAGE 131072
#define NB_BUFR (SZ_PAGE * 2 / sizeof(TUPLE))
#define NB_BUFS (SZ_PAGE * 16 / sizeof(TUPLE))
*/

#define MATCH_RATE 1

#define JT_SIZE 1200000000
#define SELECTIVITY 1000000000


typedef struct _TUPLE {
  int key;
  int val;
} TUPLE;

typedef struct _RESULT {
  int rkey;
  int rval;
  int lkey;
  int lval;
} RESULT;

typedef struct _BUCKET {
  int adr;
  int val;

} BUCKET;

