#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include "tuple.h"
#include "debug.h"


int left;
int right;

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


void
init(void)
{

  if (!(Tright = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(Tleft = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  if (!(Tjoin = (JOIN_TUPLE *)calloc(JT_SIZE, sizeof(JOIN_TUPLE)))) ERR;

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
  FILE *lp,*rp;


  //read table size from both table file
  if((lp=fopen(LEFT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }
  if(fread(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  fclose(lp);

  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  fclose(rp);
  printf("left size = %d\tright size = %d\n",left,right);


  /*******************init array*************************************/
  init();
  /*****************************************************************/


  /*全体の実行時間計測*/


  gettimeofday(&time_s,NULL);
  if((lp=fopen(LEFT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }
  if(fread(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  if(fread(Tleft,sizeof(TUPLE),left,lp)<left){
    fprintf(stderr,"file write error\n");
    exit(1);
  }

  fclose(lp);

  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  if(fread(Tright,sizeof(TUPLE),right,rp)<right){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  fclose(rp);

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
