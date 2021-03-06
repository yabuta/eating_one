#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define GRID_SIZE_Y 4
#define PART_C_NUM 1024   //cuda block size of hash partition count
#define PER_TH 30000        //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 2048      //the number of sub left tuple per one block in join and count kernel
#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 100000000   //val selectivity is 1/SELECTIVITY


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


//#define TUPLE_SIZE 8
//#define SHAREDSIZE 49152
