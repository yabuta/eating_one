#define JT_SIZE 120000000
#define SELECTIVITY 1000000000
#define MATCH_RATE 0.1  //match rate setting

#define LEFT_FILE "/home/yabuta/JoinData/non-index/left_table.out"
#define RIGHT_FILE "/home/yabuta/JoinData/non-index/right_table.out"


typedef enum {LEFT, RIGHT} LR;

typedef struct _TUPLE {
  int id;
  int val;
} TUPLE;

typedef struct _JOIN_TUPLE {
  int lval; // left value
  int rval; // right value
  int lid;
  int rid;
} JOIN_TUPLE;
