#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>
#include "debug.h"
#include "tuple.h"

#define LEFT_FILE "/home/yabuta/JoinData/hash/left_table.out"
#define RIGHT_FILE "/home/yabuta/JoinData/hash/right_table.out"

TUPLE *hrt;
int rlocation[NB_BKTENT+1];
int rcount[NB_BKTENT];

TUPLE *rt;
TUPLE *lt;
RESULT *jt;

uint right,left;

void
printDiff(struct timeval begin, struct timeval end)
{
  long diff;

  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

void diffplus(long *total,struct timeval begin,struct timeval end){
  *total += (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);

}



static int
getTupleId(void)
{
  static int id;
  
  return ++id;
}

void shuffle(TUPLE ary[],int size) {
  for(int i=0;i<size;i++){
    int j = rand()%size;
    int t = ary[i].val;
    ary[i].val = ary[j].val;
    ary[j].val = t;
  }
}


void createTuple()
{

  if (!(jt = (RESULT *)calloc(JT_SIZE, sizeof(RESULT)))) ERR;

}

void freeTuple(){

  free(rt);
  free(lt);
  free(jt);
  free(hrt);

}



int 
main(int argc,char *argv[])
{


  //RESULT result;
  int resultVal = 0;
  FILE *lp,*rp;
  struct timeval leftread_time_s, leftread_time_f;
  struct timeval rightread_time_s, rightread_time_f;
  long leftread_time = 0,rightread_time = 0;

  struct timeval begin, end,index_s,index_f,join_s,join_f;

  if(argc>1){
    printf("引数が多い\n");
    return 0;
  }

  createTuple();

  /*
  TUPLE *tlr;
  int lr;

  tlr = lt;
  lt = rt;
  rt = tlr;

  lr = left;
  left = right;
  right = lr;
  */

  //make index
  /**
     Bucket:indexの入っている配列
     Buck_array:i番目のindexのスタート位置
     idxcount:i番目のindexの数

     GPUでは配列を一つにした方が扱いやすいので
     indexとoffsetの配列を用意してある

   */

  gettimeofday(&begin, NULL);

  if((lp=fopen(LEFT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }
  if(fread(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }

  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }

  printf("left size = %d\tright size = %d\n",left,right);

  int lsize = left,rsize = right;

  if (!(lt = (TUPLE *)calloc(lsize, sizeof(TUPLE)))) ERR;
  if (!(rt = (TUPLE *)calloc(rsize, sizeof(TUPLE)))) ERR;
  if (!(hrt = (TUPLE *)calloc(rsize, sizeof(TUPLE)))) ERR;

  gettimeofday(&leftread_time_s, NULL);
  if((left=fread(lt,sizeof(TUPLE),lsize,lp))<0){
    fprintf(stderr,"file write error\n");
    exit(1);
  }

  gettimeofday(&leftread_time_f, NULL);
  diffplus(&leftread_time,leftread_time_s,leftread_time_f);

  gettimeofday(&rightread_time_s, NULL);
  if((right=fread(rt,sizeof(TUPLE),rsize,rp))<0){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  gettimeofday(&rightread_time_f, NULL);
  diffplus(&rightread_time,rightread_time_s,rightread_time_f);


  printf("file read time:\n");
  printf("Diff: %ld us (%ld ms)\n\n", leftread_time+rightread_time, (leftread_time+rightread_time)/1000);



  gettimeofday(&index_s, NULL);

  /************
   right hash
  ************/

  int idx = 0;
  for(uint i = 0; i<right ; i++){
    idx = rt[i].val % NB_BKTENT;
    rcount[idx]++;
  }

  rlocation[0] = 0;
  for(int i = 1; i<NB_BKTENT ; i++){
    rlocation[i] = rlocation[i-1] + rcount[i-1];
  }

  //for count[] reuse
  for(uint i = 0; i<NB_BKTENT ; i++){
    rcount[i] = 0;
  }

  for(uint i = 0; i<right ; i++){
    int idx = rt[i].val % NB_BKTENT;
    hrt[rlocation[idx] + rcount[idx]].key = rt[i].key;
    hrt[rlocation[idx] + rcount[idx]].val = rt[i].val;
    rcount[idx]++;
  }
  rlocation[NB_BKTENT] = right;

  gettimeofday(&index_f, NULL);

  gettimeofday(&join_s, NULL);
  /*******join******************************************************/

  int hash = 0;
  for (uint j = 0; j < left; j++) {
    hash = lt[j].val % NB_BKTENT;
    for (int i = rlocation[hash]; i < rlocation[hash+1] ;i++ ) {
      if (hrt[i].val == lt[j].val){
        jt[resultVal].rkey = hrt[i].key;
        jt[resultVal].rval = hrt[i].val;
        jt[resultVal].lkey = lt[j].key;
        jt[resultVal].lval = lt[j].val;
        resultVal++;
      }
    }
  }
  gettimeofday(&join_f, NULL);

  fclose(lp);
  fclose(rp);

  gettimeofday(&end, NULL);

  printf("*********execution time*************************\n");
  printDiff(begin, end);
  printf("\n");
  printf("*********create index time*************************\n");
  printDiff(index_s,index_f);
  printf("\n");
  printf("*********join time*************************\n");
  printDiff(join_s,join_f);
  printf("\n");
  printf("resultVal: %d\n", resultVal);
  printf("\n");

  for(uint i=0;i<3;i++){
    printf("join[%d]:left %8d \t:right: %8d\n",i,jt[i].lkey,jt[i].rkey);
    printf("left = %8d\tright = %8d\n",jt[i].lval,jt[i].rval);
    printf("\n");
  }

  freeTuple();

  return 0;
}
