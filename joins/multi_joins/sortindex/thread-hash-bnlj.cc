#include "tuple.h"
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


void
init(void)
{

  rt = (TUPLE *)calloc(right,sizeof(TUPLE));
  lt = (TUPLE *)calloc(left,sizeof(TUPLE));
  jt = (RESULT *)calloc(JT_SIZE,sizeof(RESULT));
  
  srand((unsigned)time(NULL));
  uint *used;//usedなnumberをstoreする
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  for(uint i=0; i<SELECTIVITY ;i++){
    used[i] = i;
  }
  uint selec = SELECTIVITY;

  //uniqueなnumberをvalにassignする
  for (int i = 0; i < right; i++) {
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
  for(int i=0; i<right ; i++){
    used_r[i] = i;
  }
  uint rg = right;
  uint l_diff;//
  if(MATCH_RATE != 0){
    l_diff = left/(MATCH_RATE*right);
  }else{
    l_diff = 1;
  }

  for(int i = 0; i < left; i++) {
    lt[i].key = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*right){
      uint temp = rand()%rg;
      uint temp2 = used_r[temp];
      rg = rg-1;
      used[temp] = used[rg];

      lt[i].val = rt[temp2].val;
      counter++;
    }else{

      uint temp = rand()%selec;
      uint temp2 = used[temp];
      selec = selec-1;
      used[temp] = used[selec];

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

  struct timeval sjoin,ejoin;
  long joindiff;
  gettimeofday(&sjoin, NULL);

  for(int i = 0; i<THREAD_NUM; i++){
    if (pthread_create(&(thEx[i]), NULL, executor, NULL)) ERR;

  }

  for(int i=0 ; i<THREAD_NUM; i++){
    if (pthread_join(thEx[i], NULL)) ERR;
  }

  gettimeofday(&ejoin, NULL);
  joindiff = calcDiff(sjoin, ejoin);
  
  printDiff(sjoin,ejoin);
  printf("resultVal:%d\n",resultVal);
  printf("joindiff:%ld\n",joindiff/1000);

}

int
main(int argc, char *argv[])
{

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

  init();

  TUPLE *temp;
  int tmp;
  temp = lt;
  lt = rt;
  rt = temp;
  tmp = left;
  left = right;
  right = tmp;



  long idxtime = createIndex();
  printf("index create time:%ld\n",idxtime/1000);

  createThreads();

  for(uint i=0;i<3;i++){
    printf("join[%d]:left %8d \t:right: %8d\n",i,jt[i].skey,jt[i].rkey);
    printf("left = %8d\tright = %8d\n",jt[i].sval,jt[i].rval);
    printf("\n");
  }


  return 0;
}
