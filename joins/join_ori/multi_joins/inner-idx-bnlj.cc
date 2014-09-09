#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/time.h>
#include "debug.h"

#define SZ_PAGE 4096
#define NB_BUFR (SZ_PAGE * 2 / sizeof(TUPLE))
#define NB_BUFS (SZ_PAGE * 16 / sizeof(TUPLE))

typedef struct _TUPLE {
  int key;
  int val;
} TUPLE;

typedef struct _RESULT {
  int rkey;
  int rval;
  int skey;
  int sval;
} RESULT;

typedef struct _IDX {
  int val;
  int adr;
  struct _IDX *nxt;
} IDX;

IDX Hidx;

typedef struct _HASHOBJ {
  int val;
  int adr;
  struct _HASHOBJ *nxt;
} HASHOBJ;

typedef struct _BUCKET {
  HASHOBJ head;
  HASHOBJ *tail;
} BUCKET;

BUCKET *Bucket;
#define NB_BKT_ENT 8192


void
printDiff(struct timeval begin, struct timeval end)
{
  long diff;
  
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

// create index for S
void
createIndex(void)
{
  int sfd;
  TUPLE buf[NB_BUFS];

  sfd = open("S", O_RDONLY); if (sfd == -1) ERR;
  IDX *pidx = &Hidx;
  int adr = -1; // address of tuple in the file
  while (1) {
    int ns = read(sfd, buf, sizeof(buf));
    if (ns == -1) ERR;
    else if (ns == 0) break;
    for (unsigned int i = 0; i < ns/sizeof(TUPLE); i++) {
      if (!(pidx->nxt = (IDX *)calloc(1, sizeof(IDX)))) ERR; pidx = pidx->nxt;
      pidx->val = buf[i].val;
      pidx->adr = adr;
      adr += sizeof(TUPLE);
    }
  }
  close(sfd);

  if (!(Bucket = (BUCKET *)calloc(NB_BKT_ENT, sizeof(BUCKET)))) ERR;
  for (int i = 0; i < NB_BKT_ENT; i++) Bucket[i].tail = &Bucket[i].head;
  int count = 0;
  for (pidx = Hidx.nxt; pidx; pidx = pidx->nxt) {
    int idx = pidx->val % NB_BKT_ENT;
    if (!(Bucket[idx].tail->nxt = (HASHOBJ *)calloc(1, sizeof(HASHOBJ)))) ERR;
    Bucket[idx].tail = Bucket[idx].tail->nxt;
    Bucket[idx].tail->val = pidx->val;
    Bucket[idx].tail->adr = pidx->adr;
    count++;
  }
  
  while (Hidx.nxt) {
    IDX *tmp = Hidx.nxt; Hidx.nxt = Hidx.nxt->nxt; free(tmp);
  }
}

int 
main(void)
{
  int rfd;
  int sfd;
  int nr;
  TUPLE bufR[NB_BUFR];
  RESULT result;
  int resultVal = 0;
  struct timeval begin, end;

  createIndex();

  rfd = open("R", O_RDONLY); if (rfd == -1) ERR;
  sfd = open("S", O_RDONLY); if (sfd == -1) ERR;

  gettimeofday(&begin, NULL);
  while (1) {
    nr = read(rfd, bufR, NB_BUFR * sizeof(TUPLE));
    if (nr == -1) ERR; else if (nr == 0) break;
    
    for (int i = 0; i < nr/(int)sizeof(TUPLE); i++) {
      int idx = bufR[i].val % NB_BKT_ENT;
      HASHOBJ *pho;
      for (pho = Bucket[idx].head.nxt; pho; pho = pho->nxt) {
	if (pho->val == bufR[i].val) {
	  TUPLE tpl;
	  lseek(sfd, pho->adr, SEEK_SET);
	  int ns = read(sfd, &tpl, sizeof(TUPLE));
	  if (ns == -1) ERR;
	  result.rkey = bufR[i].key;
	  result.rval = bufR[i].val;
	  result.skey = tpl.key;
	  result.sval = tpl.val;
	  resultVal += result.rval;
	}
      } 
    }
  }
  gettimeofday(&end, NULL);
  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);

  return 0;
}
