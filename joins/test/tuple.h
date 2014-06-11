

#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 512
//#define NB_BKT_ENT 16777216


/*
1048576*1048576
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 262144

4194304*4194304    
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 4194304

16777216*16777216  
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 16777216

33554432*33554432  
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 33554432

67108864*67108864  
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 67108864

*/
#define MATCH_RATE 1.0

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

/*
typedef struct _BUCKET {

    int val;
    int adr;

} BUCKET;
*/
