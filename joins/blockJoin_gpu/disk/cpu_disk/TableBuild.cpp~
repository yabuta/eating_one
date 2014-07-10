#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include "tuple.h"

#define LEFT_FILE "/home/yabuta/JoinData/non-index/left_table.out"
#define RIGHT_FILE "/home/yabuta/JoinData/non-index/right_table.out"

int left,right;

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
    int t = ary[i].val[0];
    ary[i].val[0] = ary[j].val[0];
    ary[j].val[0] = t;
  }
}

//初期化する
void
init(void)
{
  

  TUPLE *lt,*rt;
  FILE *lp,*rp;


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
    rt[i].id = getTupleId();
    uint temp = rand()%selec;
    uint temp2 = used[temp];
    selec = selec-1;
    used[temp] = used[selec];

    for(uint j= 0 ; j<VAL_NUM ; j++){
      rt[i].val[j] = temp2; 
    }
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
    lt[i].id = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*right){
      uint temp = rand()%rg;
      uint temp2 = used_r[temp];
      rg = rg-1;
      used[temp] = used[rg];

      for(uint j= 0 ; j<VAL_NUM ; j++){
        lt[i].val[j] = rt[temp2].val[j];      
      }
      counter++;
    }else{
      uint temp = rand()%selec;
      uint temp2 = used[temp];
      selec = selec-1;
      used[temp] = used[selec];
      for(uint j= 0 ; j<VAL_NUM ; j++){
        lt[i].val[j] = temp2; 
      }
    }
  }

  printf("%d\n",counter);

  
  free(used);
  free(used_r);

  shuffle(lt,left);

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


