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

void diffplus(long *total,struct timeval begin,struct timeval end){
  *total += (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);

}


/*
tableのinit.
match rateによって適合するタプルの数を決める。
match rate = the match right tuples / all right tuples

*/


void createTuple()
{
  if (!(jt = (RESULT *)calloc(JT_SIZE, sizeof(RESULT)))) ERR;

}

void freeTuple(){

  free(rt);
  free(lt);
  free(hlt);
  free(hrt);
  free(jt);

}


// create index for S
void
createPart()
{

  /************
   right hash
   ************/

#ifndef SIZEREADFILE
  for(uint i=0; i<NB_BKTENT ; i++){
    rcount[i] = 0;
    lcount[i] = 0;
  }
#endif

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
  struct timeval begin, end;
  struct timeval leftread_time_s, leftread_time_f;
  struct timeval rightread_time_s, rightread_time_f;
  struct timeval hashjoin_s, hashjoin_f;
  long leftread_time = 0,rightread_time = 0,hashjoin_time = 0;


  //read table size from both table file

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

  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }

#ifndef SIZEREADFILE
  left=LSIZE;
  right=RSIZE;
#endif
  printf("left size = %d\tright size = %d\n",left,right);

  int lsize = left,rsize = right;

  if (!(lt = (TUPLE *)calloc(lsize, sizeof(TUPLE)))) ERR;
  if (!(hlt = (TUPLE *)calloc(lsize, sizeof(TUPLE)))) ERR;
  if (!(rt = (TUPLE *)calloc(rsize, sizeof(TUPLE)))) ERR;
  if (!(hrt = (TUPLE *)calloc(rsize, sizeof(TUPLE)))) ERR;

  // Hash construction phase

  // Matching phase


  while(1){
    gettimeofday(&leftread_time_s, NULL);
    if((left=fread(lt,sizeof(TUPLE),lsize,lp))<0){
      fprintf(stderr,"file write error\n");
      exit(1);
    }
    if(left == 0) break;
    
    gettimeofday(&leftread_time_f, NULL);
    diffplus(&leftread_time,leftread_time_s,leftread_time_f);

    while(1){
      gettimeofday(&rightread_time_s, NULL);
      if((right=fread(rt,sizeof(TUPLE),rsize,rp))<0){
        fprintf(stderr,"file write error\n");
        exit(1);
      }
      if(right == 0) break;
      gettimeofday(&rightread_time_f, NULL);
      diffplus(&rightread_time,rightread_time_s,rightread_time_f);

      gettimeofday(&hashjoin_s, NULL);

      createPart();

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
      gettimeofday(&hashjoin_f, NULL);
      diffplus(&hashjoin_time,hashjoin_s,hashjoin_f);

    }
    if(fseek(rp,sizeof(int),0)!=0){
      fprintf(stderr,"file seek error.\n");
      exit(1);
    }
  }

  fclose(lp);
  fclose(rp);


  gettimeofday(&end, NULL);

  printf("all time:\n");
  printDiff(begin, end);
  printf("file read time:\n");
  printf("Diff: %ld us (%ld ms)\n", leftread_time+rightread_time, (leftread_time+rightread_time)/1000);
  printf("hash join time:\n");
  printf("Diff: %ld us (%ld ms)\n", hashjoin_time, hashjoin_time/1000);
  printf("resultVal: %d\n", resultVal);
  printf("\n\n");
  
  freeTuple();

  return 0;
}


