#define NB_BKTENT 524288 // the number of partitions


//最適hash値
// 1M * 1M   131072
// 4M * 4M   524288
// 16M * 16M 524288
// 64M * 64M 1048576


#define JT_SIZE 1200000000
#define SELECTIVITY 100000000 //

#define MATCH_RATE 0.1 //match rate setting

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
