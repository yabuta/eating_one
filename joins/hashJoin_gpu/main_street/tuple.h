#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1

#define PARTITION 64    //partition num per 1 loop
#define RADIX 6
#define PART_C_NUM 32   //cuda block size of hash partition count
#define SHARED_MAX PARTITION * PART_C_NUM
//#define LOOP 3          //partition loop times

#define LEFT_PER_TH 512
#define RIGHT_PER_TH LEFT_PER_TH       //the number of tuple per one thread of hash partition 

#define PART_STANDARD 64
#define JOIN_SHARED 256      //the number of sub left tuple per one block in join and count kernel


/*1048576 * 1048576
#define PARTITION 64    //partition num per 1 loop
#define RADIX 6
#define PART_C_NUM 32   //cuda block size of hash partition count
#define SHARED_MAX PARTITION * PART_C_NUM
#define LOOP 2          //partition loop times

#define LEFT_PER_TH 512
#define RIGHT_PER_TH 512       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 128      //the number of sub left tuple per one block in join and count kernel

*/

/* 4194304 * 4194304
#define PARTITION 64    //partition num per 1 loop
#define RADIX 6
#define PART_C_NUM 32   //cuda block size of hash partition count
#define SHARED_MAX PARTITION * PART_C_NUM
#define LOOP 3          //partition loop times

#define LEFT_PER_TH 256
#define RIGHT_PER_TH LEFT_PER_TH       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 512      //the number of sub left tuple per one block in join and count kernel
*/

/* 16777216 * 16777216
#define PARTITION 64    //partition num per 1 loop
#define RADIX 6
#define PART_C_NUM 32   //cuda block size of hash partition count
#define SHARED_MAX PARTITION * PART_C_NUM
#define LOOP 3          //partition loop times

#define LEFT_PER_TH 256
#define RIGHT_PER_TH LEFT_PER_TH       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 512      //the number of sub left tuple per one block in join and count kernel
*/

/*33554432 * 33554432
#define PARTITION 64    //partition num per 1 loop
#define RADIX 6
#define PART_C_NUM 32   //cuda block size of hash partition count
#define SHARED_MAX PARTITION * PART_C_NUM
#define LOOP 3          //partition loop times

#define LEFT_PER_TH 1024
#define RIGHT_PER_TH LEFT_PER_TH       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 512      //the number of sub left tuple per one block in join and count kernel

*/

/* 67108864 * 67108864

#define PARTITION 64    //partition num per 1 loop
#define RADIX 6
#define PART_C_NUM 32   //cuda block size of hash partition count
#define SHARED_MAX PARTITION * PART_C_NUM
#define LOOP 3          //partition loop times

#define LEFT_PER_TH 1024
#define RIGHT_PER_TH LEFT_PER_TH       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 512      //the number of sub left tuple per one block in join and count kernel

*/

/* optimatic values 10000000 10000000
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 4096       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 256      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
*/

/* parallelism optimate
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 8192       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 64      //the number of sub left tuple per one block in join and count kernel
*/

/*50000000 * 50000000 opt
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1
#define PART_C_NUM 256   //cuda block size of hash partition count
#define PER_TH 65536       //the number of tuple per one thread of hash partition 
#define B_ROW_NUM 256      //the number of sub left tuple per one block in join and count kernel
#define J_T_LEFT B_ROW_NUM/GRID_SIZE_Y
*/



#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 1000000000  //the range of random value for tuple
#define MATCH_RATE 0.1          //match rate (selectivity,%)

#define LEFT 0
#define RIGHT 1

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
