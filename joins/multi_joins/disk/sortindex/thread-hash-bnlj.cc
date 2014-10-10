#include "tuple.h"

#define LEFT_FILE "/home/yabuta/JoinData/sort-index/left_table.out"
#define RIGHT_FILE "/home/yabuta/JoinData/sort-index/right_table.out"
#define INDEX_FILE "/home/yabuta/JoinData/sort-index/index.out"


pthread_mutex_t Lk;

BUCKET *Bucket;

TUPLE *rt;
TUPLE *lt;
RESULT *jt;

int right,left;


int offset = 0;
bool finish_flag = false;

int resultVal = 0;


void
printDiff(struct timeval begin, struct timeval end)
{
  long diff;

  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

long
calcDiff(struct timeval begin, struct timeval end)
{
  long diff;
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  return diff;
}

void diffplus(long *total,struct timeval begin,struct timeval end){
  *total += (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);

}



void
init(void)
{

  jt = (RESULT *)calloc(JT_SIZE,sizeof(RESULT));
  
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

long createIndex(void)
{

  struct timeval index_s,index_e;


  gettimeofday(&index_s,NULL);

  if (!(Bucket = (BUCKET *)calloc(right, sizeof(BUCKET)))) ERR;
  for(int i=0; i<right ; i++){
    Bucket[i].val = rt[i].val;
    Bucket[i].adr = i;
  }
  qsort(0,right-1);

  for(int i=1; i<right ; i++){
    if(Bucket[i-1].val > Bucket[i].val){
      printf("sort error[%d]\n",i);
      break;
    }
  }

  gettimeofday(&index_e,NULL);
  return calcDiff(index_s,index_e);


}





LTBUFDATA LTGetter(){


  LTBUFDATA res;

  if (pthread_mutex_lock(&Lk)) ERR;
  if(finish_flag == true){
    res.size = -1;
    res.startPos = NULL;

  }else{
    if(left - offset > LT_BUF){
      res.size = LT_BUF;
      res.startPos = &(lt[offset]);
      offset += LT_BUF;
    }else{
      res.size = left-offset;
      res.startPos = &(lt[offset]);
      offset += left-offset;
      finish_flag = true;
    }
  }
  if (pthread_mutex_unlock(&Lk)) ERR;

  return res;

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


void *
executor(void *a)
{
  
  while(true){
    LTBUFDATA ltBuf = LTGetter();

    if(ltBuf.size == -1) break;
    
    /* join */
    for (int j = 0; j < ltBuf.size; j++){
      TUPLE *temp = ltBuf.startPos;
      //一致するタプルをBucketから探索し、その周辺で合致するものを探す。
      uint bidx = search(Bucket,temp[j].val,right);
      uint x = bidx;
      while(Bucket[x].val == temp[j].val){
        TUPLE rtemp,ltemp;
        rtemp = rt[Bucket[x].adr];
        ltemp = temp[j];

        if(pthread_mutex_lock(&Lk))ERR;
        jt[resultVal].skey = ltemp.key;
        jt[resultVal].sval = ltemp.val;
        jt[resultVal].rkey = rtemp.key;
        jt[resultVal].rval = rtemp.val;
        resultVal++;
        if(pthread_mutex_unlock(&Lk))ERR;
        if(x == 0) break;
        x--;
      }
      x = bidx+1;
      while(Bucket[x].val == temp[j].val){
        TUPLE rtemp,ltemp;
        rtemp = rt[Bucket[x].adr];
        ltemp = temp[j];

        if(pthread_mutex_lock(&Lk))ERR;
        jt[resultVal].skey = ltemp.key;
        jt[resultVal].sval = ltemp.val;
        jt[resultVal].rkey = rtemp.key;
        jt[resultVal].rval = rtemp.val;
        resultVal++;
        if(pthread_mutex_unlock(&Lk))ERR;
        if(x == right-1) break;
        x++;
      }

    }
  }

  return NULL; // just to complier be quiet
}


void
createThreads(void)
{
  pthread_t thEx[THREAD_NUM];
  FILE *rp,*lp,*ip;

  struct timeval begin, end;
  struct timeval sjoin,ejoin;
  struct timeval leftread_time_s, leftread_time_f;
  struct timeval rightread_time_s, rightread_time_f;
  long leftread_time = 0,rightread_time = 0;
  long joindiff;



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



  gettimeofday(&leftread_time_s, NULL);
  if((left=fread(lt,sizeof(TUPLE),lsize,lp))<0){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  gettimeofday(&leftread_time_f, NULL);
  diffplus(&leftread_time,leftread_time_s,leftread_time_f);


  gettimeofday(&sjoin, NULL);

  for(int i = 0; i<THREAD_NUM; i++){
    if (pthread_create(&(thEx[i]), NULL, executor, NULL)) ERR;

  }

  for(int i=0 ; i<THREAD_NUM; i++){
    if (pthread_join(thEx[i], NULL)) ERR;
  }

  fclose(lp);
  fclose(rp);



  gettimeofday(&ejoin, NULL);
  joindiff = calcDiff(sjoin, ejoin);


  printf("all time:\n");
  printf("join time:\n");
  printDiff(sjoin,ejoin);
  printf("file read time:\n");
  printf("Diff: %ld us (%ld ms)\n", leftread_time+rightread_time, (leftread_time+rightread_time)/1000);

  
  printf("resultVal:%d\n",resultVal);
  printf("joindiff:%ld\n",joindiff/1000);

}

int
main(int argc, char *argv[])
{

  if(argc>1){
    printf("引数が多い\n");
    return 0;
  }

  init();

  /*
  TUPLE *temp;
  int tmp;
  temp = lt;
  lt = rt;
  rt = temp;
  tmp = left;
  left = right;
  right = tmp;
  */

  /*
  long idxtime = createIndex();
  printf("index create time:%ld\n",idxtime/1000);
  */

  createThreads();

  for(uint i=0;i<3;i++){
    printf("join[%d]:left %8d \t:right: %8d\n",i,jt[i].skey,jt[i].rkey);
    printf("left = %8d\tright = %8d\n",jt[i].sval,jt[i].rval);
    printf("\n");
  }


  return 0;
}
