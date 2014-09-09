/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/


//タプルの数を設定する
#define VAL_NUM 1


typedef struct _TUPLE {
    //struct timeval t;
    int id;
    int val[VAL_NUM];

} TUPLE;

typedef struct _JOIN_TUPLE {
    //struct timeval t;
    int id;
    int lval[VAL_NUM]; // left value
    int rval[VAL_NUM]; // right value
    // the folloings are just for debug, not necessary
    int lid;
    int rid;
    //struct _JOIN_TUPLE *nxt;  
} JOIN_TUPLE;


//テーブルを表示するときのタプルの間隔。タプルが多いと大変なことになるため
#define PER_SHOW 1000000//10000000

//1blockでのスレッド数の定義。32*32=1024
#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 64
#define SCAN_SIZE_X 1024
