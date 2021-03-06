/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/

/*****
use hj.cc
 *****/

//1blockでのスレッド数の定義。
#define BLOCK_SIZE_X 1024  //outer ,left
#define BLOCK_SIZE_Y 4096  //inner ,right

#define PART 1048576

//buffer size
#define LSIZE 131072
#define RSIZE 131072
#define SIZEREADFILE

/*****
use TableBuild.cpp
 *****/

#define JT_SIZE 120000000
#define SELECTIVITY 1000000000
#define MATCH_RATE 0.1



/*****
use both
 *****/
#define VAL_NUM 1

typedef struct _TUPLE {
    int id;
    int val[VAL_NUM];

} TUPLE;

typedef struct _JOIN_TUPLE {
    int id;
    int lid;
    int lval[VAL_NUM]; // left value
    int rid;
    int rval[VAL_NUM]; // right value
} JOIN_TUPLE;


//テーブルを表示するときのタプルの間隔。タプルが多いと大変なことになるため
#define PER_SHOW 1000000//10000000


