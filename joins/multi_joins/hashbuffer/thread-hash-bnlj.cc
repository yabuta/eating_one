#include "tuple.h"

pthread_mutex_t Lk;


TUPLE *rt;
TUPLE *prt;
TUPLE *lt;
TUPLE *plt;
RESULT *jt;

int right,left;


int offset = 0;
bool finish_flag = false;

//int resultVal = 0;
int resoffset = 0;



int LL[THREAD_NUM*PARTITION];
int RL[THREAD_NUM*PARTITION];
int LLS[THREAD_NUM*PARTITION];
int RLS[THREAD_NUM*PARTITION];

int counter[THREAD_NUM+1];


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
  plt = (TUPLE *)calloc(left,sizeof(TUPLE));
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

void *lcount_partitioning(void *a)
{

  int threadIdx = *((int *)a);

  int startPos,endPos;
  if(left%THREAD_NUM != 0){
    startPos = threadIdx*(left/THREAD_NUM+1);
  }else{
    startPos = threadIdx*(left/THREAD_NUM);
  }

  if(left%THREAD_NUM != 0){
    endPos = (threadIdx+1)*(left/THREAD_NUM+1);
    if(endPos > left){
      endPos = left;
    }
  }else{
    endPos = (threadIdx+1)*(left/THREAD_NUM);
    if(endPos > left){
      endPos = left;
    }

  }

  int hash = 0;

  for(int i = startPos; i<endPos;i++){
    hash = lt[i].val % PARTITION;
    LL[hash*THREAD_NUM + threadIdx]++;
  }

}


void *lpartitioning(void *a)
{

  int threadIdx = *((int *)a);

  int startPos,endPos;
  if(left%THREAD_NUM != 0){
    startPos = threadIdx*(left/THREAD_NUM+1);
  }else{
    startPos = threadIdx*(left/THREAD_NUM);
  }

  if(left%THREAD_NUM != 0){
    endPos = (threadIdx+1)*(left/THREAD_NUM+1);
    if(endPos > left){
      endPos = left;
    }

  }else{
    endPos = (threadIdx+1)*(left/THREAD_NUM);
    if(endPos > left){
      endPos = left;
    }
  }

  int hash = 0;

  int i;
  for(i = startPos; i<endPos ;i++){
    hash = lt[i].val%PARTITION;    
    plt[LLS[hash*THREAD_NUM+threadIdx]] = lt[i];
    LLS[hash*THREAD_NUM + threadIdx]++;

  }

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



RTBUFDATA RTGetter(){


  RTBUFDATA res;

  if (pthread_mutex_lock(&Lk)) ERR;
  if(offset >= PARTITION){
    res.lsize = -1;
    res.lstartPos = NULL;
    res.rsize = -1;
    res.rstartPos = NULL;

  }else{
    if(offset == 0){
      res.lstartPos = plt;
      res.lsize = LLS[THREAD_NUM*(offset+1)-1];
      res.rstartPos = prt;
      res.rsize = RLS[THREAD_NUM*(offset+1)-1];
    }else{
      res.lstartPos = &(plt[LLS[THREAD_NUM*offset-1]]);
      res.lsize = LLS[THREAD_NUM*(offset+1)-1] - LLS[THREAD_NUM*offset-1];
      res.rstartPos = &(prt[RLS[THREAD_NUM*offset-1]]);
      res.rsize = RLS[THREAD_NUM*(offset+1)-1] - RLS[THREAD_NUM*offset-1];
    }
    offset++;
  }
  if (pthread_mutex_unlock(&Lk)) ERR;       

  return res;

}


void writeResult(RESULT *buf,int size){

  if (pthread_mutex_lock(&Lk)) ERR;
  for(int i=0; i<size ; i++){
    jt[resoffset+i] = buf[i];
  }
  resoffset += size;
  if (pthread_mutex_unlock(&Lk)) ERR;

}


void *
executor(void *a)
{

  RESULT outputbuf[BUFF_SIZE];
  int bufoffset=0;

  RTBUFDATA tBuf;

  while(true){

    tBuf = RTGetter();

    if(tBuf.lsize == -1) break;

    TUPLE *ltemp = tBuf.lstartPos;
    TUPLE *rtemp = tBuf.rstartPos;

    for (int i = 0; i < tBuf.lsize ; i++){
      for (int j=0 ; j < tBuf.rsize ; j++){        
        if (rtemp[j].val == ltemp[i].val){
          outputbuf[bufoffset].rkey = rtemp[j].key;
          outputbuf[bufoffset].rval = rtemp[j].val;
          outputbuf[bufoffset].skey = ltemp[i].key;
          outputbuf[bufoffset].sval = ltemp[i].val;
          bufoffset++;
          if(bufoffset == BUFF_SIZE){
            writeResult(outputbuf,BUFF_SIZE);
            bufoffset = 0;
          }
        }

      }
    }
  }

  writeResult(outputbuf,bufoffset);

  //printf("temp = %d\n",temp);


  return NULL; // just to complier be quiet
}


void
createThreads(void)
{
  pthread_t thEx[THREAD_NUM];
  int thid[THREAD_NUM];

  struct timeval sjoin,ejoin;
  struct timeval lp_s,lp_e,rp_s,rp_e,j_s,j_e;
  long joindiff;
  gettimeofday(&sjoin, NULL);

  printf("ok\n");

  /*partition lt */
  /*
    count partition num
    scan to caluculate start position
    move tuple
   */

  gettimeofday(&lp_s, NULL);
  for(int i = 0; i<THREAD_NUM; i++){
    thid[i] = i;
    if (pthread_create(&(thEx[i]), NULL, lcount_partitioning, &(thid[i]))) ERR;
  }

  for(int i=0 ; i<THREAD_NUM; i++){
    if (pthread_join(thEx[i], NULL)) ERR;
  }

  LLS[0] = 0;
  for(int i=1; i<THREAD_NUM*PARTITION ; i++){
    LLS[i] = LLS[i-1] + LL[i-1];
  }


  for(int i = 0; i<THREAD_NUM; i++){
    if(pthread_create(&(thEx[i]), NULL, lpartitioning, &(thid[i]))) ERR;
  }

  for(int i=0 ; i<THREAD_NUM; i++){
    if(pthread_join(thEx[i], NULL)) ERR;
  }
  gettimeofday(&lp_e, NULL);
  printDiff(lp_s,lp_e);

  /*
  int temp=0;
  for(int i=0 ; i<left ; i++){
    if(plt[i].val%PARTITION > temp){
      temp = plt[i].val%PARTITION;
    }else if(plt[i].val%PARTITION < temp){
      printf("false\t%d\t%d\t%d\n",i,plt[i].val,plt[i].val%PARTITION);
      break;
    }
  }

  exit(1);
  */
  /*
  for(int i=0 ;i<PARTITION*THREAD_NUM; i++){
    printf("LLS[%d] = %d\n",i,LLS[i]);
  }
  printf("%d\t%d\n",plt[LLS[0]+1].val,plt[LLS[0]+1].val%PARTITION);
  */

  /*partition rt*/
  gettimeofday(&rp_s, NULL);

  for(int i = 0; i<THREAD_NUM; i++){
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
  printDiff(lp_s,rp_e);


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
  printf("resultVal:%d\n",resoffset);
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
