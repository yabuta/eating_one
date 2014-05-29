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
  uint counter = 0;

  for (uint i = 0; i < right; i++) {
    rt[i].key = getTupleId();
    if(i%diff == 0 && counter < right*MATCH_RATE){
      uint temp = rand()%SELECTIVITY;
      while(used[temp] == 1) temp = rand()%SELECTIVITY;
      used[temp] = 1;
      rt[i].val = temp;
      counter++;
    }else{
      rt[i].val = SELECTIVITY + rand()%SELECTIVITY;
    }
  }
  free(used);

  counter = 0;
  for (uint i = 0; i < left; i++) {
    lt[i].key = getTupleId();
    if(i%diff == 0 && counter < right*MATCH_RATE){
      lt[i].val = rt[i].val;
      counter++;
    }else{
      lt[i].val = 2 * SELECTIVITY + rand()%SELECTIVITY;
    }
  }



}

void freeTuple(){

  free(rt);
  free(lt);
  free(jt);

}



int 
main(int argc,char *argv[])
{


  //RESULT result;
  int resultVal = 0;
  struct timeval begin, end;

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


  createTuple();

  //make index
  
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

  // join
  //gettimeofday(&begin, NULL);
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

  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);

  return 0;
}
