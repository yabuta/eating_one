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

#define LEFT_FILE "/home/yabuta/JoinData/hash/left_table.out"
#define RIGHT_FILE "/home/yabuta/JoinData/hash/right_table.out"

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
  FILE *lp,*rp;
  struct timeval begin, end ,middle;

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

  createTuple();


  gettimeofday(&begin, NULL);

  if((lp=fopen(LEFT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }
  if(fread(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  if(fread(lt,sizeof(TUPLE),left,lp)<left){
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
  if(fread(rt,sizeof(TUPLE),right,rp)<right){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  fclose(rp);

  /*
  TUPLE *temp;
  int lr;
  temp=rt;
  rt=lt;
  lt=temp;
  temp = hrt;
  hrt = hlt;
  hlt = temp;


  lr = left;
  left = right;
  right = lr;
  */

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


