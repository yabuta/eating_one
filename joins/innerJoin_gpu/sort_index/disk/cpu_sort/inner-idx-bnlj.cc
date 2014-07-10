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

#define LEFT_FILE "/home/yabuta/JoinData/sort-index/left_table.out"
#define RIGHT_FILE "/home/yabuta/JoinData/sort-index/right_table.out"
#define INDEX_FILE "/home/yabuta/JoinData/sort-index/index.out"

BUCKET *Bucket;

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


void createTuple()
{

  if (!(rt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(lt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  if (!(jt = (RESULT *)calloc(JT_SIZE, sizeof(RESULT)))) ERR;

}


void freeTuple(){

  free(rt);
  free(lt);
  free(jt);

}


/***
binary search

**/

uint search(BUCKET *b,int num,uint right)
{
  int m,l,r;
  l=0;
  r=right-1;
  do{
    m=(l+r)/2;
    if(num < b[m].val)r=m-1;else l=m+1;
  }while(l<=r&&num!=b[m].val);

  return m;
}


int 
main(int argc,char *argv[])
{


  //RESULT result;
  FILE *lp,*rp,*ip;
  int resultVal = 0;
  struct timeval begin, end;
  //  struct timeval s_s,s_f,w_s,w_f;
  //  long sea=0,wri=0;


  /*tuple and index init******************************************/  

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
  if((ip=fopen(INDEX_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }
  Bucket = (BUCKET *)malloc(right*sizeof(BUCKET));
  if(fread(Bucket,sizeof(BUCKET),right,ip)<right){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  fclose(lp);


  // join
  gettimeofday(&begin, NULL);

  for (uint j = 0; j < left; j++) {

    //一致するタプルをBucketから探索し、その周辺で合致するものを探す。
    uint bidx = search(Bucket,lt[j].val,right);
    uint x = bidx;
    while(Bucket[x].val == lt[j].val){
      jt[resultVal].lkey = lt[j].key;
      jt[resultVal].lval = lt[j].val;
      jt[resultVal].rkey = rt[Bucket[x].adr].key;
      jt[resultVal].rval = rt[Bucket[x].adr].val;      
      resultVal++;
      if(x == 0) break;
      x--;
    }
    x = bidx+1;
    while(Bucket[x].val == lt[j].val){
      jt[resultVal].lkey = lt[j].key;
      jt[resultVal].lval = lt[j].val;
      jt[resultVal].rkey = rt[Bucket[x].adr].key;
      jt[resultVal].rval = rt[Bucket[x].adr].val;      
      resultVal++;
      if(x == right-1) break;
      x++;
    }

  }
  gettimeofday(&end, NULL);

  //printf("search time = %ldms\nwrite time = %ldms\n",sea/1000,wri/1000);

  printf("*******execution time****************\n");
  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);
  printf("\n");


  for(uint i=0;i<3&&i<(uint)resultVal;i++){
    printf("join[%d]:left %8d \t:right: %8d\n",i,jt[i].lkey,jt[i].rkey);
    printf("left = %8d\tright = %8d\n",jt[i].lval,jt[i].rval);
    printf("\n");
  }

  freeTuple();

  return 0;
}
