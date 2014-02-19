#define SZ_PAGE 4096
#define NB_BUF  (SZ_PAGE * 16 / sizeof(TUPLE))
#define NB_BKTENT 4 // the number of partitions
#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 32
#define PART_C_NUM 64
#define TUPLE_SIZE 8
#define SHAREDSIZE 49152
#define B_ROW_NUM 1024
//#define B_ROW_NUM (SHAREDSIZE/TUPLE_SIZE)
#define PER_TH 100

#define NUM_VAL 1

#define JT_SIZE 120000000
#define SELECTIVITY 100


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
