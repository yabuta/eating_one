#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include "debug.h"
#include "tuple.h"


#define LEFT_FILE "/home/yabuta/JoinData/hash-index/left_table.out"
#define RIGHT_FILE "/home/yabuta/JoinData/hash-index/right_table.out"
#define INDEX_FILE "/home/yabuta/JoinData/hash-index/index.out"

int left,right;

BUCKET *Bucket;
int Buck_array[NB_BKT_ENT];
int idxcount[NB_BKT_ENT];


static int
getTupleId(void)
{
  static int id;
  
  return ++id;
}

//shuffle tuple
void shuffle(TUPLE ary[],int size) {    
  srand((unsigned)time(NULL));
  for(int i=0;i<size;i++){
    int j = rand()%size;
    TUPLE t = ary[i];
    ary[i] = ary[j];
    ary[j] = t;
  }
}

// create index for S
void
createIndex(TUPLE *rt)
{

  int count=0;
  for (int i = 0; i < NB_BKT_ENT; i++) idxcount[i] = 0;
  for(uint i=0 ; i<right ; i++){
    int idx = rt[i].val % NB_BKT_ENT;
    idxcount[idx]++;
    count++;
  }

  count = 0;

  if (!(Bucket = (BUCKET *)calloc(right, sizeof(BUCKET)))) exit(1);
  for (int i = 0; i < NB_BKT_ENT; i++) {
    if(idxcount[i] == 0){
      Buck_array[i] = -1;
    }else{
      Buck_array[i] = count;
    }
    count += idxcount[i];
  }
  for (int i = 0; i < NB_BKT_ENT; i++) idxcount[i] = 0;
  //for (pidx = Hidx.nxt; pidx; pidx = pidx->nxt) {
  for(uint i=0; i<right ; i++){
    int idx = rt[i].val % NB_BKT_ENT;
    Bucket[Buck_array[idx] + idxcount[idx]].val = rt[i].val;
    Bucket[Buck_array[idx] + idxcount[idx]].adr = i;
    idxcount[idx]++;
  }

}

//初期化する
void
init(void)
{
  

  TUPLE *lt,*rt;
  FILE *lp,*rp,*ip;


  //メモリ割り当てを行う
  //タプルに初期値を代入

  lt = (TUPLE *)calloc(left,sizeof(TUPLE));
  rt = (TUPLE *)calloc(right,sizeof(TUPLE));

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
  for (uint i = 0; i < left ; i++) {
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

  createIndex(rt);

  if((lp=fopen(LEFT_FILE,"w"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }

  if(fwrite(&left,sizeof(int),1,lp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  if(fwrite(lt,sizeof(TUPLE),left,lp)<left){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  fclose(lp);


  if((rp=fopen(RIGHT_FILE,"w"))==NULL){
    fprintf(stderr,"file open error(right)\n");
    exit(1);
  }
  if(fwrite(&right,sizeof(int),1,rp)<1){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  if(fwrite(rt,sizeof(TUPLE),right,rp)<right){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  fclose(rp);

  if((ip=fopen(INDEX_FILE,"w"))==NULL){
    fprintf(stderr,"file open error(left)\n");
    exit(1);
  }
  if(fwrite(Bucket,sizeof(BUCKET),right,ip)<right){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  if(fwrite(Buck_array,sizeof(int),NB_BKT_ENT,ip)<NB_BKT_ENT){
    fprintf(stderr,"file write error\n");
    exit(1);
  }
  if(fwrite(idxcount,sizeof(int),NB_BKT_ENT,ip)<NB_BKT_ENT){
    fprintf(stderr,"file write error\n");
    exit(1);
  }

  fclose(lp);

}


int main(int argc, char *argv[]){

  if(argc<4){
    if(argv[1]==NULL){
      printf("argument1 is nothing.\n");
      exit(1);
    }else{
      right=atoi(argv[1]);
      printf("right num :%d\n",right);
    }
    
    if(argv[2]==NULL){
      printf("argument2 is nothing.\n");
      exit(1);
    }else{
      left=atoi(argv[2]);
      printf("left num :%d\n",left);
    }
  }




  printf("init starting...\n");
  init();
  printf("...init finish\n");

}


