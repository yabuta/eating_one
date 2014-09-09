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

void diffplus(long *total,struct timeval begin,struct timeval end){
  *total += (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);

}



void createTuple()
{

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
  struct timeval leftread_time_s, leftread_time_f;
  struct timeval rightread_time_s, rightread_time_f;
  struct timeval join_s,join_f;
  long leftread_time = 0,rightread_time = 0, join_time = 0;

  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
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
  fclose(ip);


  createTuple();

  gettimeofday(&begin, NULL);


  /*tuple and index init******************************************/  
  if((lp=fopen(LEFT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }

  if(fread(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }

#ifndef SIZEREADFILE
  left = LSIZE;
#endif
  int lsize = left;


  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }

  printf("left size = %d\tright size = %d\n",left,right);

  if (!(lt = (TUPLE *)calloc(lsize, sizeof(TUPLE)))) ERR;
  if (!(rt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;

  gettimeofday(&rightread_time_s, NULL);
  if(fread(rt,sizeof(TUPLE),right,rp)<right){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  gettimeofday(&rightread_time_f, NULL);
  diffplus(&rightread_time,rightread_time_s,rightread_time_f);

  // join
  while(1){
    gettimeofday(&leftread_time_s, NULL);
    if((left=fread(lt,sizeof(TUPLE),lsize,lp))<0){
      fprintf(stderr,"file write error\n");
      exit(1);
    }
    if(left == 0) break;
    gettimeofday(&leftread_time_f, NULL);
    diffplus(&leftread_time,leftread_time_s,leftread_time_f);

    gettimeofday(&join_s, NULL);
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
    gettimeofday(&join_f, NULL);
    diffplus(&join_time,join_s,join_f);
  }


  fclose(lp);
  fclose(rp);

  gettimeofday(&end, NULL);

  //printf("search time = %ldms\nwrite time = %ldms\n",sea/1000,wri/1000);

  printf("*******execution time****************\n");
  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);
  printf("\n");
  printf("file read time:\n");
  printf("Diff: %ld us (%ld ms)\n", leftread_time+rightread_time, (leftread_time+rightread_time)/1000);
  printf("left table file read time:\n");
  printf("Diff: %ld us (%ld ms)\n", leftread_time, leftread_time/1000);
  printf("right table file read time:\n");
  printDiff(rightread_time_s,rightread_time_f);
  printf("join time:\n");
  printf("Diff: %ld us (%ld ms)\n", join_time, join_time/1000);
  printf("\n\n");

  printf("result size: %d\n", resultVal);
  printf("\n");


  for(uint i=0;i<3&&i<(uint)resultVal;i++){
    printf("join[%d]:left %8d \t:right: %8d\n",i,jt[i].lkey,jt[i].rkey);
    printf("left = %8d\tright = %8d\n",jt[i].lval,jt[i].rval);
    printf("\n");
  }

  freeTuple();

  return 0;
}
