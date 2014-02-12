#define VAL_NUM 1

typedef struct _TUPLE {
    struct timeval t;
  int id;
  int val[VAL_NUM];
    //struct _TUPLE *nxt;  
} TUPLE;

typedef struct _JOIN_TUPLE {
    struct timeval t;
  int id;
  int lval[VAL_NUM]; // left value
  int rval[VAL_NUM]; // right value
  // the folloings are just for debug, not necessary
  int lid;
  int rid;
    //struct _JOIN_TUPLE *nxt;  
} JOIN_TUPLE;

//引数をとる場合のコードを書きたかったが、GPUに渡すのめんどくさいので途中蜂起

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
