/******
use hj.cc
 ******/
#define BLOCK_SIZE_X 256

#define LSIZE 524288
#define SIZEREADFILE

/******
use TableBuild.cpp
 ******/
#define MATCH_RATE 0.1
#define JT_SIZE 120000000
#define SELECTIVITY 1000000000

/******
use both
 ******/
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
    int val;
    int adr;

} BUCKET;
