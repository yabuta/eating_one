#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/time.h>
#include "debug.h"
#include "tuple.h"

TUPLE *rt;
TUPLE *lt;
RESULT *jt;

TUPLE *hrt;
TUPLE *hlt;

int rlocation[NB_BKTENT];
int rcount[NB_BKTENT];
int llocation[NB_BKTENT];
int lcount[NB_BKTENT];

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
    p->val = n; // selectivity = 0.01 or to be 1 million tuple of jt
  }

}

void createTuple()
{

  if (!(rt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(lt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  if (!(hrt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(hlt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
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


// create index for S
void
createPart()
{

  /*
  int fd;
  int *bfdAry;
  char partFile[BUFSIZ];
  TUPLE buf[NB_BUF];
  */

  for(int i = 0; i<right ; i++){
    int idx = rt[i].val % NB_BKTENT;
    rcount[idx]++;
  }

  rlocation[0] = 0;
  for(int i = 1; i<NB_BKTENT ; i++){
    rlocation[i] = rlocation[i-1] + rcount[i-1];
  }

  //for count[] reuse
  for(int i = 0; i<NB_BKTENT ; i++){
    rcount[i] = 0;
  }

  for(int i = 0; i<right ; i++){
    int idx = rt[i].val % NB_BKTENT;
    hrt[rlocation[idx] + rcount[idx]].key = rt[i].key; 
    hrt[rlocation[idx] + rcount[idx]].val = rt[i].val; 
    rcount[idx]++;

  }
  

  for(int i = 0; i<left ; i++){
    int idx = lt[i].val % NB_BKTENT;
    lcount[idx]++;
  }

  llocation[0] = 0;
  for(int i = 1; i<NB_BKTENT ; i++){
    llocation[i] = llocation[i-1] + lcount[i-1];
  }

  //for count[] reuse
  for(int i = 0; i<NB_BKTENT ; i++){
    lcount[i] = 0;
  }

  for(int i = 0; i<left ; i++){
    int idx = lt[i].val % NB_BKTENT;
    hlt[llocation[idx] + lcount[idx]].key = lt[i].key; 
    hlt[llocation[idx] + lcount[idx]].val = lt[i].val; 
    lcount[idx]++;

  }
  
}

int
openPart(const char *partFile, int id)
{
  int fd;
  char buf[BUFSIZ];

  bzero(buf, sizeof(buf));
  sprintf(buf, "hash-part-%s-%d", partFile, id);
  fd = open(buf, O_RDONLY);
  if (fd == -1) ERR;

  return fd;
}

int 
main(int argc,char *argv[])
{

  RESULT result;
  int resultVal = 0;
  struct timeval begin, end ,middle;

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


  gettimeofday(&begin, NULL);
  // Hash construction phase
  createPart();


  /*
  for (int i = 0; i < NB_BKTENT; i++) {
    printf("%d\t%d\t%d\t%d\n",rlocation[i],rcount[i],llocation[i],lcount[i]);

  }
  */

  // Matching phase
  gettimeofday(&middle, NULL);

  for (int i = 0; i < NB_BKTENT; i++) {

    for (int j = 0; j < rcount[i]; j++) {
      for (int k = 0; k < lcount[i]; k++) {
        if (hrt[rlocation[i] + j].val == hlt[llocation[i] + k].val) {
          result.rkey = hrt[rlocation[i] + j].key;
          result.rval = hrt[rlocation[i] + j].val;
          result.skey = hlt[llocation[i] + k].key;
          result.sval = hlt[llocation[i] + k].val;
          resultVal++;
        }
      }
    }
  }

  gettimeofday(&end, NULL);

  printDiff(begin, end);
  printDiff(begin, middle);
  printDiff(middle, end);
  printf("resultVal: %d\n", resultVal);

  
  freeTuple();

  return 0;
}


