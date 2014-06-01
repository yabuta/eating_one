#define NB_BUCKET 134217728 //radix

#define MATCH_RATE 1.00  //match rate setting

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

