typedef struct _TUPLE {
  struct timeval t;
  int id;
  int val;
  struct _TUPLE *nxt;  
} TUPLE;

typedef struct _JOIN_TUPLE {
  struct timeval t;
  int id;
  int lval; // left value
  int rval; // right value
  // the folloings are just for debug, not necessary
  int lid;
  int rid;
  struct _JOIN_TUPLE *nxt;  
} JOIN_TUPLE;

//引数をとる場合のコードを書きたかったが、GPUに渡すのめんどくさいので途中蜂起
#define ARG

#define MAX_LEFT 10
#define MAX_RIGHT 10

#define PER_SHOW 1000
