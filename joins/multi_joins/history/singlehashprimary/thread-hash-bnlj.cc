#include "tuple.h"

pthread_mutex_t Lk;


TUPLE *rt;
TUPLE *prt;
TUPLE *lt;
RESULT *jt;

int right,left;

int resultVal = 0;

int RL[THREAD_NUM*PARTITION];
int RLS[THREAD_NUM*PARTITION];


int offset = 0;
bool finish_flag = false;


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
  prt = (TUPLE *)calloc(right,sizeof(TUPLE));
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
  shuffle(rt,right);

}


void *rcount_partitioning(void *a)
{

  int threadIdx = *((int *)a);

  int startPos,endPos;
  if(right%THREAD_NUM != 0){
    startPos = threadIdx*(right/THREAD_NUM+1);
  }else{
    startPos = threadIdx*(right/THREAD_NUM);
  }

  if(right%THREAD_NUM != 0){
    endPos = (threadIdx+1)*(right/THREAD_NUM+1);
    if(endPos > right){
      endPos = right;
    }
  }else{
    endPos = (threadIdx+1)*(right/THREAD_NUM);
    if(endPos > right){
      endPos = right;
    }
  }


  int hash = 0;

  for(int i = startPos; i<endPos;i++){
    hash = rt[i].val % PARTITION;
    RL[hash*THREAD_NUM + threadIdx]++;
  }

}


void *rpartitioning(void *a)
{

  int threadIdx = *((int *)a);

  int startPos,endPos;
  if(right%THREAD_NUM != 0){
    startPos = threadIdx*(right/THREAD_NUM+1);
  }else{
    startPos = threadIdx*(right/THREAD_NUM);
  }

  if(right%THREAD_NUM != 0){
    endPos = (threadIdx+1)*(right/THREAD_NUM+1);
    if(endPos > right){
      endPos = right;
    }
  }else{
    endPos = (threadIdx+1)*(right/THREAD_NUM);
    if(endPos > right){
      endPos = right;
    }
  }

  int hash = 0;

  for(int i = startPos; i<endPos ;i++){
    hash = rt[i].val%PARTITION;
    prt[RLS[hash*THREAD_NUM+threadIdx]] = rt[i];
    RLS[hash*THREAD_NUM + threadIdx]++;    

  }

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



void *
executor(void *a)
{

  /* join */
  while(true){

    LTBUFDATA ltBuf = LTGetter();
    if(ltBuf.size == -1) break;
    
    TUPLE *ltemp = ltBuf.startPos;
    for (int i = 0; i < ltBuf.size ; i++){
      int hashval = ltemp[i].val%PARTITION;
      if(hashval==0){
        for (int j=0 ; j<RLS[THREAD_NUM*(hashval+1)-1] ; j++){
          if (prt[j].val == ltemp[i].val){
            TUPLE rtemp = prt[j];
            if (pthread_mutex_lock(&Lk)) ERR;
            jt[resultVal].rkey = ltemp[i].key;
            jt[resultVal].rval = ltemp[i].val;
            jt[resultVal].skey = rtemp.key;
            jt[resultVal].sval = rtemp.val;
            resultVal++;
            if (pthread_mutex_unlock(&Lk)) ERR;
          }
        }
      }else{
        for (int j=RLS[THREAD_NUM*hashval-1] ; j<RLS[THREAD_NUM*(hashval+1)-1] ; j++){
          if (prt[j].val == ltemp[i].val){
            TUPLE rtemp = prt[j];
            if (pthread_mutex_lock(&Lk)) ERR;
            jt[resultVal].rkey = ltemp[i].key;
            jt[resultVal].rval = ltemp[i].val;
            jt[resultVal].skey = rtemp.key;
            jt[resultVal].sval = rtemp.val;
            resultVal++;
            if (pthread_mutex_unlock(&Lk)) ERR;
          }
        }

      }
    }
  }

  //printf("temp = %d\n",temp);


  return NULL; // just to complier be quiet
}


void
createThreads(void)
{
  pthread_t thEx[THREAD_NUM];
  int thid[THREAD_NUM];

  struct timeval sjoin,ejoin;
  struct timeval rp_s,rp_e,j_s,j_e;
  long joindiff;
  gettimeofday(&sjoin, NULL);


  /*partition rt*/
  gettimeofday(&rp_s, NULL);

  for(int i = 0; i<THREAD_NUM; i++){
    thid[i] = i;
    if (pthread_create(&(thEx[i]), NULL, rcount_partitioning, &(thid[i]))) ERR;
  }

  for(int i=0 ; i<THREAD_NUM; i++){
    if (pthread_join(thEx[i], NULL)) ERR;
  }


  RLS[0] = 0;
  for(int i=1; i<THREAD_NUM*PARTITION ; i++){
    RLS[i] = RLS[i-1] + RL[i-1];
  }


  for(int i = 0; i<THREAD_NUM; i++){
    if (pthread_create(&(thEx[i]), NULL, rpartitioning, &(thid[i]))) ERR;
  }

  for(int i=0 ; i<THREAD_NUM; i++){
    if (pthread_join(thEx[i], NULL)) ERR;
  }

  gettimeofday(&rp_e,NULL);
  printDiff(rp_s,rp_e);


  /*
  int temp=0;
  for(int i=0 ; i<right ; i++){
    if(prt[i].val%PARTITION > temp){
      temp = prt[i].val%PARTITION;
    }else if(prt[i].val%PARTITION < temp){
      printf("false\t%d\t%d\t%d\n",i,prt[i].val,prt[i].val%PARTITION);
      break;
    }
  }

  exit(1);
  */


  /*join*/
  gettimeofday(&j_s,NULL);
  for(int i = 0; i<THREAD_NUM; i++){
    if (pthread_create(&(thEx[i]), NULL, executor, NULL)) ERR;
  }

  for(int i=0 ; i<THREAD_NUM; i++){
    if (pthread_join(thEx[i], NULL)) ERR;
  }

  gettimeofday(&j_e,NULL);
  printDiff(j_s,j_e);

  gettimeofday(&ejoin, NULL);
  joindiff = calcDiff(sjoin, ejoin);
  
  printDiff(sjoin,ejoin);
  printf("resultVal:%d\n",resultVal);
  printf("joindiff:%ld\n",joindiff/1000);

  for(uint i=0;i<3;i++){
    printf("join[%d]:left %8d \t:right: %8d\n",i,jt[i].skey,jt[i].rkey);
    printf("left = %8d\tright = %8d\n",jt[i].sval,jt[i].rval);
    printf("\n");
  }


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
  createThreads();

  return 0;
}
