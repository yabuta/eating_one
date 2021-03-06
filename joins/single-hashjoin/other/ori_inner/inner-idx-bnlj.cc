#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include "debug.h"

#define SZ_PAGE 4096
#define NB_BUFR (SZ_PAGE * 2 / sizeof(TUPLE))
#define NB_BUFS (SZ_PAGE * 16 / sizeof(TUPLE))
#define NUM_VAL 1

int right,left;

typedef struct _TUPLE {
  int key;
  int val[NUM_VAL];
} TUPLE;

typedef struct _RESULT {
  int rkey;
  int rval;
  int skey;
  int sval;
} RESULT;

typedef struct _IDX {
  int val;
  int adr;
  struct _IDX *nxt;
} IDX;

IDX Hidx;

typedef struct _HASHOBJ {
  int val;
  int adr;
  struct _HASHOBJ *nxt;
} HASHOBJ;

typedef struct _BUCKET {
  HASHOBJ head;
  HASHOBJ *tail;
} BUCKET;

BUCKET *Bucket;
#define NB_BKT_ENT 8192

TUPLE *rt;
TUPLE *lt;
RESULT *jt;


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
    p->val[i] = n; // selectivity = 1.0
  }

}

void createTuple()
{

  if (!(rt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(lt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  if (!(jt = (RESULT *)calloc(right * left, sizeof(RESULT)))) ERR;

  srand((unsigned)time(NULL));
  for (int i = 0; i < right; i++) {
    getTuple(&(rt[i]),rand()%100);
  }

  for (int i = 0; i < left; i++) {
    getTuple(&(lt[i]),rand()%100);
  }


}

void freeTuple(){

  free(rt);
  free(lt);
  free(jt);

}


// create index for S
void
createIndex(void)
{

  IDX *pidx = &Hidx;
  //int adr = -1; // address of tuple in the file
  for (int i = 0; i < left; i++) {
    if (!(pidx->nxt = (IDX *)calloc(1, sizeof(IDX)))) ERR; pidx = pidx->nxt;
    pidx->val = lt[i].val[0];
    pidx->adr = i;
  }

  if (!(Bucket = (BUCKET *)calloc(NB_BKT_ENT, sizeof(BUCKET)))) ERR;
  for (int i = 0; i < NB_BKT_ENT; i++) Bucket[i].tail = &Bucket[i].head;
  int count=0;
  for (pidx = Hidx.nxt; pidx; pidx = pidx->nxt) {
    int idx = pidx->val % NB_BKT_ENT;
    if (!(Bucket[idx].tail->nxt = (HASHOBJ *)calloc(1, sizeof(HASHOBJ)))) ERR;
    Bucket[idx].tail = Bucket[idx].tail->nxt;
    Bucket[idx].tail->val = pidx->val;
    Bucket[idx].tail->adr = pidx->adr;
    count++;
  }

  printf("%d\n",count);
  
  while (Hidx.nxt) {
    IDX *tmp = Hidx.nxt; Hidx.nxt = Hidx.nxt->nxt; free(tmp);
  }
}

int 
main(int argc,char *argv[])
{

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
  createIndex();


  gettimeofday(&begin, NULL);
  for (int i = 0; i < right; i++) {
    int idx = rt[i].val[0] % NB_BKT_ENT;
    HASHOBJ *pho;
    for (pho = Bucket[idx].head.nxt; pho; pho = pho->nxt) {
      if (pho->val == rt[i].val[0]) {
        jt[resultVal].rkey = rt[i].key;
        jt[resultVal].rval = rt[i].val[0];
        jt[resultVal].skey = lt[pho->adr].key;
        jt[resultVal].sval = lt[pho->adr].val[0];
        resultVal++;
      } 
    }
  }
  gettimeofday(&end, NULL);
  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);

  return 0;
}
