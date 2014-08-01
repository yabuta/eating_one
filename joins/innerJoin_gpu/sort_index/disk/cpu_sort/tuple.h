/******
use hj.cc
 ******/
#define LSIZE 8388608
#define SIZEREADFILE

/******
use TableBuild.cpp
 ******/
#define MATCH_RATE 0.1
#define JT_SIZE 1200000000
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
  int adr;
  int val;

} BUCKET;

