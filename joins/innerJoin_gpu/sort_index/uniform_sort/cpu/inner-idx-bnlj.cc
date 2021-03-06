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

void shuffle(TUPLE ary[],int size) {
  for(int i=0;i<size;i++){
    int j = rand()%size;
    int t = ary[i].val;
    ary[i].val = ary[j].val;
    ary[j].val = t;
  }
}


void createTuple()
{

  if (!(rt = (TUPLE *)calloc(right, sizeof(TUPLE)))) ERR;
  if (!(lt = (TUPLE *)calloc(left, sizeof(TUPLE)))) ERR;
  if (!(jt = (RESULT *)calloc(JT_SIZE, sizeof(RESULT)))) ERR;


  srand((unsigned)time(NULL));
  uint *used;//usedなnumberをstoreする
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  for(uint i=0; i<SELECTIVITY ;i++){
    used[i] = i;
  }
  uint selec = SELECTIVITY;

  //uniqueなnumberをvalにassignする
  for (uint i = 0; i < right; i++) {
    if(&(rt[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }
    rt[i].key = getTupleId();
    uint temp = rand()%selec;
    uint temp2 = used[temp];
    selec = selec-1;
    used[temp] = used[selec];

    rt[i].val = temp2; 

  }


  uint counter = 0;//matchするtupleをcountする。
  uint *used_r;
  used_r = (uint *)calloc(right,sizeof(uint));
  for(uint i=0; i<right ; i++){
    used_r[i] = i;
  }
  uint rg = right;
  uint l_diff;//
  if(MATCH_RATE != 0){
    l_diff = left/(MATCH_RATE*right);
  }else{
    l_diff = 1;
  }
  uint temp = rand()%selec;
  uint temp2 = used[temp];
  selec = selec-1;
  used[temp] = used[selec];

  for (uint i = 0; i < left; i++) {
    lt[i].key = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*right){
      /*
      uint temp = rand()%rg;
      uint temp2 = used_r[temp];
      rg = rg-1;
      used[temp] = used[rg];
      */
      lt[i].val = rt[0].val;      
      counter++;
    }else{
      lt[i].val = temp2; 
      
    }
  }

  printf("%d\n",counter);

  
  free(used);
  free(used_r);

  shuffle(lt,left);

}


/***************create index*****************************/
/*
  swap(TUPLE *a,TUPLE *b)
  qsort(int p,int q)
  createIndex(void)
*/

void swap(BUCKET *a,BUCKET *b)
{
  BUCKET temp;
  temp = *a;
  *a=*b;
  *b=temp;
}

void qsort(int p,int q)
{
  int i,j;
  int pivot;

  i = p;
  j = q;

  pivot = Bucket[(p+q)/2].val;

  while(1){
    while(Bucket[i].val < pivot) i++;
    while(pivot < Bucket[j].val) j--;
    if(i>=j) break;
    swap(&Bucket[i],&Bucket[j]);
    i++;
    j--;
  }
  if(p < i-1) qsort(p,i-1);
  if(j+1 < q) qsort(j+1,q);

}

void createIndex(void)
{

  if (!(Bucket = (BUCKET *)calloc(right, sizeof(BUCKET)))) ERR;
  for(uint i=0; i<right ; i++){
    Bucket[i].val = rt[i].val;
    Bucket[i].adr = i;
  }
  qsort(0,right-1);

  for(uint i=1; i<right ; i++){
    if(Bucket[i-1].val > Bucket[i].val){
      printf("sort error[%d]\n",i);
      break;
    }
  }

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
  int resultVal = 0;
  struct timeval begin, end,index_s,index_f;
  //  struct timeval s_s,s_f,w_s,w_f;
  //  long sea=0,wri=0;

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

  //make index
  gettimeofday(&index_s, NULL);
  createIndex();
  gettimeofday(&index_f, NULL);


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
  printf("******index create time*************\n");
  printDiff(index_s,index_f);
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
