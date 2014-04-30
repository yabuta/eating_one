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

int right,left;

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

static void getTuple(TUPLE *p,int n)
{

  int i;

  p->key = getTupleId();
  for(i=0; i<NUM_VAL; i++){
    p->val = n; // selectivity = 0.01
  }

}

void createTuple()
{

  if (!(rt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(lt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  //if (!(jt = (RESULT *)calloc((right * left)/50, sizeof(RESULT)))) ERR;
  if (!(jt = (RESULT *)calloc(JT_SIZE, sizeof(RESULT)))) ERR;

  srand((unsigned)time(NULL));
  for (int i = 0; i < right; i++) {
    getTuple(&(rt[i]),rand()%SELECTIVITY);
  }

  for (int i = 0; i < left; i++) {
    getTuple(&(lt[i]),rand()%SELECTIVITY);
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

  gettimeofday(&begin, NULL);

  createTuple();

  //make index
  
  int count = 0;
  for (unsigned int i = 0; i < NB_BUCKET; i++) idxcount[i] = 0;

  for (int i = 0; i < left; i++) {
    int idx = lt[i].val % NB_BUCKET;
    idxcount[idx]++;
    count++;
  }

  //test
    
  count = 0;

  if (!(Bucket = (BUCKET *)calloc(left, sizeof(BUCKET)))) ERR;
  for (unsigned int i = 0; i < NB_BUCKET; i++) {
    if(idxcount[i] == 0){
      Buck_array[i] = -1;
    }else{
      Buck_array[i] = count;
    }
    count += idxcount[i];

  }
    

  for (unsigned int i = 0; i < NB_BUCKET; i++) idxcount[i] = 0;
  for (int i = 0; i < left; i++) {
    int idx = lt[i].val % NB_BUCKET;
    Bucket[Buck_array[idx] + idxcount[idx]].val = lt[i].val;
    Bucket[Buck_array[idx] + idxcount[idx]].adr = i;
    idxcount[idx]++;
  }

  // join
  //gettimeofday(&begin, NULL);

  for (int j = 0; j < right; j++) {
    int hash = rt[j].val % NB_BUCKET;
    for (int i = 0; i < idxcount[hash] ;i++ ) {
      if (Bucket[Buck_array[hash] + i].val == rt[j].val) {
        jt[resultVal].skey = lt[Bucket[Buck_array[hash] + i].adr].key;
        jt[resultVal].sval = lt[Bucket[Buck_array[hash] + i].adr].val;
        jt[resultVal].rkey = rt[j].key;
        jt[resultVal].rval = rt[j].val;
        resultVal++;
        //printf("%d\t%d\n",Bucket[Buck_array[hash] + i].val,lt[j].val);
      }
    }
  }
  gettimeofday(&end, NULL);

  /*
  for(int i = 0;i<resultVal;i++){

    if(i%100000 == 0){
      printf("lid=%d\tlval=%d\trid=%d\trval=%d\n",jt[i].skey,jt[i].sval,jt[i].rkey,jt[i].rval);
    }

  }
  */

  freeTuple();

  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);

  return 0;
}
