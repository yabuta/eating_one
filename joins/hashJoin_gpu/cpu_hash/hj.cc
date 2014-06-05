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


/*
tableのinit.
match rateによって適合するタプルの数を決める。
match rate = the match right tuples / all right tuples

*/

void createTuple()
{

  if (!(rt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(lt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  if (!(hrt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(hlt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
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

  /*left init*/
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
      used[temp] = 1;
      lt[i].val = temp;
    }
  }
  free(used);
  
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


  /************
   right hash
   ************/

  for(uint i = 0; i<right ; i++){
    int idx = rt[i].val % NB_BKTENT;
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
  

  /********
   left hash
   *******/

  for(uint i = 0; i<left ; i++){
    int idx = lt[i].val % NB_BKTENT;
    lcount[idx]++;
  }

  llocation[0] = 0;
  for(int i = 1; i<NB_BKTENT ; i++){
    llocation[i] = llocation[i-1] + lcount[i-1];
  }

  //for count[] reuse
  for(uint i = 0; i<NB_BKTENT ; i++){
    lcount[i] = 0;
  }

  for(uint i = 0; i<left ; i++){
    int idx = lt[i].val % NB_BKTENT;
    hlt[llocation[idx] + lcount[idx]].key = lt[i].key; 
    hlt[llocation[idx] + lcount[idx]].val = lt[i].val; 
    lcount[idx]++;

  }
  
}

int 
main(int argc,char *argv[])
{

  RESULT result[JT_SIZE];
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

  // Matching phase
  gettimeofday(&middle, NULL);

  for (int i = 0; i < NB_BKTENT; i++) {
    for (int j = 0; j < lcount[i]; j++) {
      for (int k = 0; k < rcount[i]; k++) {
        if (hlt[llocation[i] + j].val == hrt[rlocation[i] + k].val) {
          result[resultVal].rkey = hrt[llocation[i] + j].key;
          result[resultVal].rval = hrt[llocation[i] + j].val;
          result[resultVal].skey = hlt[rlocation[i] + k].key;
          result[resultVal].sval = hlt[rlocation[i] + k].val;
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


