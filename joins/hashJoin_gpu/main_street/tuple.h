#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1

#define PARTITION 64    //partition num per 1 loop
#define RADIX 6
#define PART_C_NUM 16   //cuda block size of hash partition count
#define SHARED_MAX PARTITION * PART_C_NUM
//#define LOOP 3          //partition loop times

#define LEFT_PER_TH 256
#define RIGHT_PER_TH LEFT_PER_TH       //the number of tuple per one thread of hash partition 

#define PART_STANDARD 64
#define JOIN_SHARED 256      //the number of sub left tuple per one block in join and count kernel

/*optimization setting
#define BLOCK_SIZE_X 1024 //cuda block size of join and count kernel 
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_Y 1

#define PARTITION 64    //partition num per 1 loop
#define RADIX 6
#define PART_C_NUM 16   //cuda block size of hash partition count
#define SHARED_MAX PARTITION * PART_C_NUM
//#define LOOP 3          //partition loop times

#define LEFT_PER_TH 256
#define RIGHT_PER_TH LEFT_PER_TH       //the number of tuple per one thread of hash partition 

#define PART_STANDARD 64
#define JOIN_SHARED 256      //the number of sub left tuple per one block in join and count kernel



*/


#define JT_SIZE 120000000 //max result tuple size
#define SELECTIVITY 1000000000  //the range of random value for tuple
#define MATCH_RATE 0.1          //match rate (selectivity,%)

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

