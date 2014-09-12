/******
use hj.cc
 ******/
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 67108864

/*
1048576*1048576
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 262144

4194304*4194304    
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 4194304

8388608*8388608
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 8388608

16777216*16777216  
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 16777216

33554432*33554432  
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 33554432

67108864*67108864  
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 67108864

134217728*134217728
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 67108864
*/

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
