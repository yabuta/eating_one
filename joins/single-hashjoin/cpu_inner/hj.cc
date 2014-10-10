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

  if (!(rt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(lt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  if (!(jt = (RESULT *)calloc(JT_SIZE, sizeof(RESULT)))) ERR;


  srand((unsigned)time(NULL));
  uint *used;//usedなnumberをstoreする
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  for(uint i=0; i<SELECTIVITY ;i++){
    used[i] = i;
  }
  uint selec = SELECTIVITY;

  //uniqueなnumberをvalにassignする
  for (uint i = 0; i < right; i++) {
    if(&(rt[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }
    rt[i].key = getTupleId();
    uint temp = rand()%selec;
    uint temp2 = used[temp];
    selec = selec-1;
    used[temp] = used[selec];

    rt[i].val = temp2; 

  }

  uint counter = 0;//matchするtupleをcountする。
  uint *used_r;
  used_r = (uint *)calloc(right,sizeof(uint));
  for(uint i=0; i<right ; i++){
    used_r[i] = i;
  }
  uint rg = right;
  uint l_diff;//
  if(MATCH_RATE != 0){
    l_diff = left/(MATCH_RATE*right);
  }else{
    l_diff = 1;
  }

  for (uint i = 0; i < left; i++) {
    lt[i].key = getTupleId();

    if(i%l_diff == 0 && counter < MATCH_RATE*right){
      uint temp = rand()%rg;
      uint temp2 = used_r[temp];
      rg = rg-1;
      used[temp] = used[rg];
      
      lt[i].val = rt[temp2].val;      
      counter++;
    }else{
      uint temp = rand()%selec;
      uint temp2 = used[temp];
      selec = selec-1;
      used[temp] = used[selec];
      lt[i].val = temp2; 

    }
  }

  printf("%d\n",counter);

  
  free(used);
  free(used_r);

  shuffle(lt,left);

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
  struct timeval begin, end,index_s,index_f,join_s,join_f;

  if(argc>3){
    printf("引数が多い\n");
    return 0;
  }else if(argc<3){
    printf("引数が足りない\n");
    return 0;
  }else{
    left=atoi(argv[1]);
    right=atoi(argv[2]);

    printf("left=%d:right=%d\n",left,right);
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

  gettimeofday(&index_s, NULL);

  /************
   right hash
  ************/

  if (!(hrt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;  

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
