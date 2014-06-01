#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include "debug.h"

#define JT_SIZE 120000000
#define SELECTIVITY 10000000
#define MATCH_RATE 1.00  //match rate setting

int left;
int right;

typedef enum {LEFT, RIGHT} LR;

typedef struct _TUPLE {
  int id;
  int val;
} TUPLE;

typedef struct _JOIN_TUPLE {
  int lval; // left value
  int rval; // right value
  // the folloings are just for debug, not necessary
  int lid;
  int rid;
} JOIN_TUPLE;

TUPLE *Tright;
TUPLE *Tleft;
JOIN_TUPLE *Tjoin;

static int
getTupleId(void)
{
  static int id;
  
  return ++id;
}


static void
getJoinTuple(TUPLE *lt, TUPLE *rt,JOIN_TUPLE *jt)
{
  int i;

  jt->lval = lt->val;
  jt->rval = rt->val;  
  
  // lid & rid are just for debug
  jt->lid = lt->id;
  jt->rid = rt->id;

  
}


void
join()
{

  int i,j;
  int count=0;
  for (i = 0 ; i < left ; i++) {
    for (j = 0 ; j < right ; j++) {
      if(Tleft[i].val == Tright[j].val){
        getJoinTuple(&(Tleft[i]),&(Tright[j]),&(Tjoin[count]));
        count++;
      }
    }
  }
  printf("%d\n",count);

}


/*
tableのinit.
match rateによって適合するタプルの数を決める。
match rate = the match right tuples / all right tuples

*/

void
init(void)
{

  if (!(Tright = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(Tleft = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  if (!(Tjoin = (JOIN_TUPLE *)calloc(JT_SIZE, sizeof(JOIN_TUPLE)))) ERR;

  srand((unsigned)time(NULL));
  uint *used;
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  uint diff;

  if(MATCH_RATE != 0){
    diff = 1/MATCH_RATE;
  }else{
    diff = 1;
  }
  uint counter = 0;

  for (uint i = 0; i < right; i++) {
    Tright[i].id = getTupleId();
    if(i%diff == 0 && counter < MATCH_RATE*right){
      uint temp = rand()%SELECTIVITY;
      while(used[temp] == 1) temp = rand()%SELECTIVITY;
      used[temp] = 1;
      Tright[i].val = temp;
      counter++;
    }else{
      Tright[i].val = SELECTIVITY + rand()%SELECTIVITY;
    }
  }
  free(used);

  counter = 0;
  uint l_diff;
  if(MATCH_RATE != 0){
    l_diff = left/(MATCH_RATE*right);
  }else{
    l_diff = 1;
  }
  for (uint i = 0; i < left; i++) {
    Tleft[i].id = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*right){
      Tleft[i].val = Tright[counter*diff].val;
      counter++;
    }else{
      Tleft[i].val = 2 * SELECTIVITY + rand()%SELECTIVITY;
    }
  }
}

//メモリ解放のため新しく追加した関数。バグがあるかも
void
tuple_free(void){


  free(Tleft);
  free(Tright);
  free(Tjoin);

}


int
main(int argc, char *argv[])
{

  struct timeval time_s,time_f;
  double time_cal;
  int i;

  if(argc>3){
    printf("引数が多い\n");
    return 0;
  }else if(argc<2){
    printf("引数が足りない\n");
    return 0;
  }else{
    left=atoi(argv[1]);
    right=atoi(argv[2]);

    printf("left=%d:right=%d\n",left,right);
  }

  init();

  gettimeofday(&time_s,NULL);
  join();
  gettimeofday(&time_f,NULL);

  time_cal=(time_f.tv_sec-time_s.tv_sec)*1000*1000+(time_f.tv_usec-time_s.tv_usec);
  printf("Caluculation with Devise    : %6f(miri sec)\n",time_cal/1000);

  printf("%d %d\n",sizeof(TUPLE),sizeof(JOIN_TUPLE));


  //結果の表示　むやみにやったら死ぬので気を付ける
  /*
  for(int i=0;i < right*left ;i++){
    if(i%100000==0){
      printf("id = %d\n",Tjoin[i].id);
      for(int j=0;j<VAL_NUM;j++){
        printf("lval[%d] = %d\trval[%d] = %d\n",j,Tjoin[i].lval[j],j,Tjoin[i].rval[j]);
        
      }
    }

  }
  */

  //割り当てたメモリを開放する
  tuple_free();

  return 0;
}
