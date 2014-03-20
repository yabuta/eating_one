#define SZ_PAGE 4096
#define NB_BUF  (SZ_PAGE * 16 / sizeof(TUPLE))
#define NB_BKTENT 1000000// the number of partitions

#define JT_SIZE 1200000000
#define SELECTIVITY 100000000


#define NUM_VAL 1


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
