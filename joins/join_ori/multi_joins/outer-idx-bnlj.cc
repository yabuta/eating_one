#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>
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

typedef struct _HASHOBJ {
  TUPLE tuple;
  struct _HASHOBJ *nxt;
} HASHOBJ;

typedef struct _BUCKET {
  HASHOBJ head;
  HASHOBJ *tail;
} BUCKET;

#define NB_BUCKET NB_BUFR


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

int 
main(void)
{
  int rfd;
  int sfd;
  int nr;
  int ns;
  TUPLE bufR[NB_BUFR];
  TUPLE bufS[NB_BUFS];
  RESULT result;
  int resultVal = 0;
  struct timeval begin, end;
  BUCKET bucket[NB_BUCKET];

  gettimeofday(&begin, NULL);
  rfd = open("R", O_RDONLY); if (rfd == -1) ERR;
  sfd = open("S", O_RDONLY); if (sfd == -1) ERR;
  bzero(bucket, sizeof(BUCKET) * NB_BUCKET);

  int cnt = 0;
  long iodiff = 0;
  long joindiff = 0;

  while (1) {
    nr = read(rfd, bufR, NB_BUFR * sizeof(TUPLE));
    if (nr == -1) ERR; else if (nr == 0) break;
    
    for (unsigned int i = 0; i < NB_BUCKET; i++) {
      while (bucket[i].head.nxt) {
	HASHOBJ *tmp = bucket[i].head.nxt;
	bucket[i].head.nxt = bucket[i].head.nxt->nxt;
	free(tmp);
      }
      bucket[i].tail = &bucket[i].head;
    }

    for (int i = 0; i < nr/(int)sizeof(TUPLE); i++) {
      int hkey = bufR[i].val % NB_BUCKET;
      if (!(bucket[hkey].tail->nxt = (HASHOBJ *)calloc(1, sizeof(HASHOBJ)))) ERR;
      bucket[hkey].tail = bucket[hkey].tail->nxt;
      bucket[hkey].tail->tuple = bufR[i];
    }


    while (1) {
      struct timeval bio, eio;
      struct timeval bjoin, ejoin;

      gettimeofday(&bio, NULL);
      ns = read(sfd, bufS, NB_BUFS * sizeof(TUPLE));
      if (ns == -1) ERR; else if (ns == 0) break;
      gettimeofday(&eio, NULL);
      iodiff += calcDiff(bio, eio);

      cnt++;

      // join
      gettimeofday(&bjoin, NULL);
      for (int j = 0; j < ns/(int)sizeof(TUPLE); j++) {
	int hash = bufS[j].val % NB_BUCKET;
	for (HASHOBJ *o = bucket[hash].head.nxt; o; o = o->nxt) {
	  if (o->tuple.val == bufS[j].val) {
	    result.rkey = o->tuple.key;
	    result.rval = o->tuple.val;
	    result.skey = bufS[j].key;
	    result.sval = bufS[j].val;
	    resultVal += result.rval;
	  }
	}
      }
      gettimeofday(&ejoin, NULL);
      joindiff += calcDiff(bjoin, ejoin);
    }
  }
  gettimeofday(&end, NULL);


  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);
  printf("iodiff: %ld\n", iodiff);
  printf("joindiff: %ld\n", joindiff);

  return 0;
}
