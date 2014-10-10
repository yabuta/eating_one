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

void diffplus(long *total,struct timeval begin,struct timeval end){
  *total += (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);

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

  jt = (RESULT *)calloc(JT_SIZE,sizeof(RESULT));

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
        if (temp[j].val == lt[i].val){
          if (pthread_mutex_lock(&Lk)) ERR;       
          jt[resultVal].rkey = lt[i].key;
          jt[resultVal].rval = lt[i].val;
          jt[resultVal].skey = temp[j].key;
          jt[resultVal].sval = temp[j].val;
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

  FILE *lp,*rp;

  struct timeval leftread_time_s, leftread_time_f;
  struct timeval rightread_time_s, rightread_time_f;
  struct timeval sjoin,ejoin;
  struct timeval begin,end;
  long leftread_time = 0,rightread_time = 0;
  long joindiff;

  //read table size from both table file
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

  int lsize = left,rsize = right;

  printf("lsize = %d\t rsize = %d\n",lsize,rsize);

  gettimeofday(&begin, NULL);

  lt = (TUPLE *)malloc(lsize*sizeof(TUPLE));
  rt = (TUPLE *)malloc(rsize*sizeof(TUPLE));

  gettimeofday(&leftread_time_s, NULL);
  if((left=fread(lt,sizeof(TUPLE),lsize,lp))<0){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  gettimeofday(&leftread_time_f, NULL);
  diffplus(&leftread_time,leftread_time_s,leftread_time_f);

  gettimeofday(&rightread_time_s, NULL);
  if((right=fread(rt,sizeof(TUPLE),rsize,rp))<0){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  gettimeofday(&rightread_time_f, NULL);
  diffplus(&rightread_time,rightread_time_s,rightread_time_f);


  gettimeofday(&sjoin, NULL);

  for(int i = 0; i<THREAD_NUM; i++){
    if (pthread_create(&(thEx[i]), NULL, executor, NULL)) ERR;

  }

  for(int i=0 ; i<THREAD_NUM; i++){
    if (pthread_join(thEx[i], NULL)) ERR;
  }

  gettimeofday(&ejoin, NULL);

  fclose(lp);
  fclose(rp);

  gettimeofday(&end, NULL);

  joindiff = calcDiff(sjoin, ejoin);

  printf("all time:\n");
  printDiff(begin,end);
  printf("join time:\n");
  printDiff(sjoin,ejoin);
  printf("file read time:\n");
  printf("Diff: %ld us (%ld ms)\n", leftread_time+rightread_time, (leftread_time+rightread_time)/1000);


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

  if(argc>1){
    printf("引数が多い\n");
    return 0;
  }

  init();
  createThreads();

  return 0;
}
