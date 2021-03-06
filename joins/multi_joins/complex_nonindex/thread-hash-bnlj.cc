#include "tuple.h"

#define BUFF_SIZE 1024

pthread_mutex_t Lk;


TUPLE *rt;
TUPLE *lt;
RESULT *jt;

int right,left;


int offset = 0;
bool finish_flag = false;

//int resoffset = 0;
int resultVal=0;

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
    for(int k=0 ; k<VAL_NUM ; k++){
      double t = ary[i].val[k];
      ary[i].val[k] = ary[j].val[k];
      ary[j].val[k] = t;
    }
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

    for(int j=0; j<VAL_NUM; j++){
      rt[i].val[j] = temp2;
    }

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

      for(int j=0; j<VAL_NUM;j++){
        lt[i].val[j] = rt[temp2].val[j];
      }
      counter++;
    }else{

      uint temp = rand()%selec;
      uint temp2 = used[temp];
      selec = selec-1;
      used[temp] = used[selec];

      for(int j=0 ;j<VAL_NUM ; j++){
        lt[i].val[j] = temp2;
      }

    }
  }

  printf("%d\n",counter);


  free(used);
  free(used_r);

  shuffle(lt,left);
  shuffle(rt,right);

}


RTBUFDATA RTGetter(){


  RTBUFDATA res;

  if (pthread_mutex_lock(&Lk)) ERR;
  if(finish_flag == true){
    res.size = -1;
    res.startPos = NULL;

  }else{
    if(right - offset > RT_BUF){
      res.size = RT_BUF;
      res.startPos = &(rt[offset]);
      offset += RT_BUF;
    }else{
      res.size = right-offset;
      res.startPos = &(rt[offset]);
      offset += right-offset;
      finish_flag = true;
    }
  }
  if (pthread_mutex_unlock(&Lk)) ERR;       

  return res;

}

bool eval(TUPLE lt,TUPLE rt){

  double temp = 0;
  double temp2 = 0;
  for(int i=0 ; i<VAL_NUM ; i++){
    temp2 = rt.val[i]-lt.val[i];
    temp += temp2*temp2;
  }

  return temp < DISTANCE * DISTANCE;


}




void *
executor(void *a)
{

  RTBUFDATA rtBuf;

  TUPLE *temp;    

  while(true){
    rtBuf = RTGetter();
    
    if(rtBuf.size == -1) break;
    
    /* join */
    for (int i = 0; i < left; i++) {
      temp = rtBuf.startPos;
      for (int j=0 ; j<rtBuf.size ; j++) {
        if (eval(temp[j],lt[i])){
          if (pthread_mutex_lock(&Lk)) ERR;       
          jt[resultVal].rkey = lt[i].key;
          jt[resultVal].skey = temp[j].key;
          for(int k=0 ; k<VAL_NUM ; k++){
            jt[resultVal].rval[k] = lt[i].val[k];
            jt[resultVal].sval[k] = temp[j].val[k];
          }
          resultVal++;
          if (pthread_mutex_unlock(&Lk)) ERR;
        }
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
  createThreads();

  return 0;
}
