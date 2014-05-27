#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 4096       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 256      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 100000000  //val selectivity is 1/SELECTIVITY
#define RES_MAX 1000000
#define MATCH_RATE 10

/*1048576 * 1048576
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 512       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 64      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 250000  //val selectivity is 1/SELECTIVITY
*/

/* 4194304 * 4194304
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 2048       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 128      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 100000000  //val selectivity is 1/SELECTIVITY
#define RES_MAX 1000000
*/


/* optimatic values 10000000 10000000
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 4096       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 256      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 100000000  //val selectivity is 1/SELECTIVITY
#define RES_MAX 1000000
*/

/* parallelism optimate
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 8192       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 64      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 169000000  //val selectivity is 1/SELECTIVITY
*/

/*50000000 * 50000000 opt
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 65536       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 256      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 169000000  //val selectivity is 1/SELECTIVITY
#define RES_MAX 1000000
*/

/* 16777216 * 16777216
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 8192       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 128      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 100000000  //val selectivity is 1/SELECTIVITY
#define RES_MAX 1000000
*/

/* 67108864 * 67108864
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 65536       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 256      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 169000000  //val selectivity is 1/SELECTIVITY
#define RES_MAX 1000000
*/

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
