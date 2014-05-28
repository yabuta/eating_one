/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/


#define JT_SIZE 120000000
#define SELECTIVITY 100000000
#define MATCH_RATE 3/100


typedef struct _TUPLE {
    //struct timeval t;
    int id;
    int val;

} TUPLE;

typedef struct _JOIN_TUPLE {
    //struct timeval t;
    int id;
    int lval; // left value
    int rval; // right value
    // the folloings are just for debug, not necessary
    int lid;
    int rid;
    //struct _JOIN_TUPLE *nxt;  
} JOIN_TUPLE;


//テーブルを表示するときのタプルの間隔。タプルが多いと大変なことになるため
#define PER_SHOW 1000000//10000000

//1blockでのスレッド数の定義。32*32=1024
#define BLOCK_SIZE_X 4096
#define BLOCK_SIZE_Y 512

