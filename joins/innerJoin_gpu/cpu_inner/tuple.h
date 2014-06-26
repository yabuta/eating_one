#define NB_BUCKET 134217728 //radix

#define MATCH_RATE 0.1  //match rate setting

#define JT_SIZE 1200000000
#define SELECTIVITY 1000000000


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

