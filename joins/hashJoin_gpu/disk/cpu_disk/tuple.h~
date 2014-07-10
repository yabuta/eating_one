#define NB_BKTENT 524288 // the number of partitions


//最適hash値
// 1M * 1M   131072
// 4M * 4M   524288
// 16M * 16M 524288
// 32M * 32M 1048576
// 64M * 64M 1048576
// 128M * 128M 2097152

#define JT_SIZE 1200000000
#define SELECTIVITY 1000000000 //

#define MATCH_RATE 1 //match rate setting

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
