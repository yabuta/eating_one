//#define SZ_PAGE 40960
//#define NB_BUFR (SZ_PAGE * 2 / sizeof(TUPLE))
//#define NB_BUFS (SZ_PAGE * 16 / sizeof(TUPLE))
#define NUM_VAL 1
#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 512
#define NB_BKT_ENT 16777216

#define MATCH_RATE 10/100

#define JT_SIZE 120000000
#define SELECTIVITY 100000000


int right,left;

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

typedef struct _IDX {
  int val;
  int adr;
  struct _IDX *nxt;
} IDX;

IDX Hidx;

/*
typedef struct _HASHOBJ {
  int val;
  int adr;
    //struct _HASHOBJ *nxt;
} HASHOBJ;
*/

typedef struct _BUCKET {

    int val;
    int adr;

} BUCKET;
