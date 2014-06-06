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


BUCKET *Bucket;
int Buck_array[NB_BUCKET];
int idxcount[NB_BUCKET];

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


void createTuple()
{

  if (!(rt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(lt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  //if (!(jt = (RESULT *)calloc((right * left)/50, sizeof(RESULT)))) ERR;
  if (!(jt = (RESULT *)calloc(JT_SIZE, sizeof(RESULT)))) ERR;

  srand((unsigned)time(NULL));
  uint *used;
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  uint diff;

  if(MATCH_RATE != 0){
    diff = 1/MATCH_RATE;
  }else{
    diff = 1;
  }

  for (uint i = 0; i < right; i++) {
    rt[i].key = getTupleId();
    uint temp = rand()%SELECTIVITY;
    while(used[temp] == 1) temp = rand()%SELECTIVITY;
    used[temp] = 1;
    rt[i].val = temp;
  }

  uint counter = 0;
  uint l_diff;
  if(MATCH_RATE != 0){
    l_diff = left/(MATCH_RATE*right);
  }else{
    l_diff = 1;
  }
  for (uint i = 0; i < left; i++) {
    lt[i].key = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*right){
      lt[i].val = rt[counter*diff].val;
      counter++;
    }else{
      uint temp = rand()%SELECTIVITY;
      while(used[temp] == 1) temp = rand()%SELECTIVITY;
      lt[i].val = temp;
    }
  }
  free(used);
  

}

void freeTuple(){

  free(rt);
  free(lt);
  free(jt);
  free(Bucket);

}



int 
main(int argc,char *argv[])
{


  //RESULT result;
  int resultVal = 0;
  struct timeval begin, end,index_s,index_f;

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

  //make index
  /**
     Bucket:indexの入っている配列
     Buck_array:i番目のindexのスタート位置
     idxcount:i番目のindexの数

     GPUでは配列を一つにした方が扱いやすいので
     indexとoffsetの配列を用意してある

   */


  gettimeofday(&index_s, NULL);
  int count = 0;
  for (unsigned int i = 0; i < NB_BUCKET; i++) idxcount[i] = 0;
  for (uint i = 0; i < right; i++) {
    int idx = rt[i].val % NB_BUCKET;
    idxcount[idx]++;
    //count++;
  }
  count = 0;
  if (!(Bucket = (BUCKET *)calloc(right, sizeof(BUCKET)))) ERR;
  for (unsigned int i = 0; i < NB_BUCKET; i++) {
    if(idxcount[i] == 0){
      Buck_array[i] = -1;
    }else{
      Buck_array[i] = count;
    }
    count += idxcount[i];
  }
  for (unsigned int i = 0; i < NB_BUCKET; i++) idxcount[i] = 0;
  for (uint i = 0; i < right; i++) {
    int idx = rt[i].val % NB_BUCKET;
    Bucket[Buck_array[idx] + idxcount[idx]].val = rt[i].val;
    Bucket[Buck_array[idx] + idxcount[idx]].adr = i;
    idxcount[idx]++;
  }
  gettimeofday(&index_f, NULL);



  /*******join******************************************************/
  gettimeofday(&begin, NULL);

  for (uint j = 0; j < left; j++) {
    int hash = lt[j].val % NB_BUCKET;
    for (int i = 0; i < idxcount[hash] ;i++ ) {
      if (Bucket[Buck_array[hash] + i].val == lt[j].val) {
        jt[resultVal].skey = rt[Bucket[Buck_array[hash] + i].adr].key;
        jt[resultVal].sval = rt[Bucket[Buck_array[hash] + i].adr].val;
        jt[resultVal].rkey = lt[j].key;
        jt[resultVal].rval = lt[j].val;
        resultVal++;
      }
    }
  }
  gettimeofday(&end, NULL);

  freeTuple();

  printf("*********create index time*************************\n");
  printDiff(index_s,index_f);
  printf("\n");
  printf("*********execution time*************************\n");
  printDiff(begin, end);
  printf("\n");
  printf("resultVal: %d\n", resultVal);
  printf("\n");

  return 0;
}
