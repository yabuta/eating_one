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

#define LEFT_FILE "/home/yabuta/JoinData/hash-index/cpu/left_table.out"
#define RIGHT_FILE "/home/yabuta/JoinData/hash-index/cpu/right_table.out"
#define INDEX_FILE "/home/yabuta/JoinData/hash-index/cpu/index.out"

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
  free(Bucket);

}



int 
main(int argc,char *argv[])
{


  //RESULT result;
  FILE *lp,*rp,*ip;
  int resultVal = 0;
  struct timeval begin, end;


  //read table size from both table file
  if((lp=fopen(LEFT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }
  if(fread(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file read(lsize) error\n");
    exit(1);
  }
  fclose(lp);

  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file read(rsize) error\n");
    exit(1);
  }
  fclose(rp);

  printf("left size = %d\tright size = %d\n",left,right);


  //memory allocate
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

  /*全体の実行時間計測*/
  gettimeofday(&begin, NULL);


  if((lp=fopen(LEFT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }
  if(fread(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file read(lsize) error\n");
    exit(1);
  }
  if(fread(lt,sizeof(TUPLE),left,lp)<left){
    fprintf(stderr,"file read(lt) error\n");
    exit(1);
  }
  fclose(lp);

  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file read(rsize) error\n");
    exit(1);
  }
  if(fread(rt,sizeof(TUPLE),right,rp)<right){
    fprintf(stderr,"file read(rt) error\n");
    exit(1);
  }
  fclose(rp);


  //read index
  /**
     Bucket:indexの入っている配列
     Buck_array:i番目のindexのスタート位置
     idxcount:i番目のindexの数

     GPUでは配列を一つにした方が扱いやすいので
     indexとoffsetの配列を用意してある

   */

  if((ip=fopen(INDEX_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(index)\n");
    exit(1);
  }
  Bucket = (BUCKET *)calloc(right ,sizeof(BUCKET));

  if(fread(Bucket,sizeof(BUCKET),right,ip)<right){
    fprintf(stderr,"file read(BUCKET) error\n");
    exit(1);
  }
  if(fread(Buck_array,sizeof(int),NB_BUCKET,ip)<NB_BUCKET){
    fprintf(stderr,"file read(Buck_array) error\n");
    exit(1);
  }
  if(fread(idxcount,sizeof(int),NB_BUCKET,ip)<NB_BUCKET){
    fprintf(stderr,"file read(idxcount) error\n");
    exit(1);
  }
  fclose(lp);


  /*

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

  */
  /*******join******************************************************/
  gettimeofday(&begin, NULL);

  for (uint j = 0; j < left; j++) {
    int hash = lt[j].val % NB_BUCKET;
    for (int i = 0; i < idxcount[hash] ;i++ ) {
      if (Bucket[Buck_array[hash] + i].val == lt[j].val) {
        jt[resultVal].rkey = rt[Bucket[Buck_array[hash] + i].adr].key;
        jt[resultVal].rval = rt[Bucket[Buck_array[hash] + i].adr].val;
        jt[resultVal].lkey = lt[j].key;
        jt[resultVal].lval = lt[j].val;
        resultVal++;
      }
    }
  }
  gettimeofday(&end, NULL);

  printf("*********execution time*************************\n");
  printDiff(begin, end);
  printf("\n");
  printf("resultVal: %d\n", resultVal);
  printf("\n");

  for(uint i=0;i<3;i++){
    printf("join[%d]:left %8d \t:right: %8d\n",i,jt[i].lkey,jt[i].rkey);
    printf("left = %8d\tright = %8d\n",jt[i].lval,jt[i].rval);
    printf("\n");
  }

  freeTuple();

  return 0;
}
