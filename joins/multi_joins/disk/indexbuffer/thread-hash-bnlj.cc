#include "tuple.h"

#define BUFF_SIZE 1024

#define LEFT_FILE "/home/yabuta/JoinData/hash-index/cpu/left_table.out"
#define RIGHT_FILE "/home/yabuta/JoinData/hash-index/cpu/right_table.out"
#define INDEX_FILE "/home/yabuta/JoinData/hash-index/cpu/index.out"

pthread_mutex_t Lk;

BUCKET *Bucket;
int Buck_array[NB_BUCKET];
int idxcount[NB_BUCKET];

TUPLE *rt;
TUPLE *lt;
RESULT *jt;

int right,left;


int offset = 0;
bool finish_flag = false;

int resoffset = 0;

/*
int counter[THREAD_NUM];
int scan[THREAD_NUM];
*/

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

  jt = (RESULT *)calloc(JT_SIZE,sizeof(RESULT));

}

long createIndex(){

  struct timeval index_s,index_f;

  gettimeofday(&index_s, NULL);
  int count = 0;
  for (unsigned int i = 0; i < NB_BUCKET; i++) idxcount[i] = 0;
  for (int i = 0; i < right; i++) {
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
  for (int i = 0; i < right; i++) {
    int idx = rt[i].val % NB_BUCKET;
    Bucket[Buck_array[idx] + idxcount[idx]].val = rt[i].val;
    Bucket[Buck_array[idx] + idxcount[idx]].adr = i;
    idxcount[idx]++;
  }
  gettimeofday(&index_f, NULL);

  return calcDiff(index_s,index_f);


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

  LTBUFDATA ltBuf;
  TUPLE *ltemp;

  while(true){

    ltBuf = LTGetter();

    if(ltBuf.size == -1) break;

    ltemp = ltBuf.startPos;

    /* join */
    for (int j = 0; j < ltBuf.size; j++){
      int hash = ltemp[j].val % NB_BUCKET;
      for (int i = 0; i < idxcount[hash] ;i++ ){
        if (Bucket[Buck_array[hash] + i].val == ltemp[j].val) {
          outputbuf[bufoffset].rkey = rt[Bucket[Buck_array[hash]+i].adr].key;
          outputbuf[bufoffset].rval = rt[Bucket[Buck_array[hash]+i].adr].val;
          outputbuf[bufoffset].skey = ltemp[j].key;
          outputbuf[bufoffset].sval = ltemp[j].val;
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


  return NULL; // just to complier be quiet
}


void
createThreads(void)
{
  pthread_t thEx[THREAD_NUM];
  FILE *lp,*rp,*ip;
  struct timeval leftread_time_s, leftread_time_f;
  struct timeval rightread_time_s, rightread_time_f;
  struct timeval sjoin,ejoin;
  struct timeval begin,end;

  long leftread_time = 0,rightread_time = 0;
  long joindiff;


  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file read(rsize) error\n");
    exit(1);
  }
  if((ip=fopen(INDEX_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(index)\n");
    exit(1);
  }
  Bucket = (BUCKET *)malloc(right*sizeof(BUCKET));

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

  fclose(rp);
  fclose(ip);


  if((lp=fopen(LEFT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }

  if(fread(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file read(lsize) error\n");
    exit(1);
  }
  fclose(lp);


  /*全体の実行時間計測*/
  gettimeofday(&begin, NULL);

  //read table size from both table file
  if((lp=fopen(LEFT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }

  if(fread(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file read(lsize) error\n");
    exit(1);
  }

  int lsize = left;

  if((rp=fopen(RIGHT_FILE,"r"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fread(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file read(rsize) error\n");
    exit(1);
  }

  printf("left size = %d\tright size = %d\n",left,right);


  if (!(lt = (TUPLE *)malloc(lsize*sizeof(TUPLE)))) ERR;
  if (!(rt = (TUPLE *)malloc(right*sizeof(TUPLE)))) ERR;


  gettimeofday(&rightread_time_s, NULL);
  if(fread(rt,sizeof(TUPLE),right,rp)<right){
    fprintf(stderr,"file read(rt) error\n");
    exit(1);
  }
  gettimeofday(&rightread_time_f, NULL);
  diffplus(&rightread_time,rightread_time_s,rightread_time_f);

  gettimeofday(&leftread_time_s, NULL);
  if((left=fread(lt,sizeof(TUPLE),lsize,lp))<0){
    fprintf(stderr,"file read(lt) error\n");
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
  gettimeofday(&ejoin, NULL);

  fclose(lp);
  fclose(rp);


  gettimeofday(&end, NULL);


  joindiff = calcDiff(sjoin, ejoin);

  printf("all time:\n");
  printDiff(begin,end);
  printf("file read time:\n");
  printf("Diff: %ld us (%ld ms)\n", leftread_time+rightread_time, (leftread_time+rightread_time)/1000);
  printf("join time:\n");
  printDiff(sjoin,ejoin);

  printf("resultVal:%d\n",resoffset);
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
