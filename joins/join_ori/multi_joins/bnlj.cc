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

void
printDiff(struct timeval begin, struct timeval end)
{
  long diff;

  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
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

  rfd = open("R", O_RDONLY); if (rfd == -1) ERR;
  sfd = open("S", O_RDONLY); if (sfd == -1) ERR;

  gettimeofday(&begin, NULL);
  while (1) {
    nr = read(rfd, bufR, NB_BUFR * sizeof(TUPLE));
    if (nr == -1) ERR; else if (nr == 0) break;

    while (1) {
      struct timeval hoge, geho;
      gettimeofday(&hoge, NULL);
      ns = read(sfd, bufS, NB_BUFS * sizeof(TUPLE));
      gettimeofday(&geho, NULL);
      //printDiff(hoge, geho);
      if (ns == -1) ERR; else if (ns == 0) break;

      // join
      for (int i = 0; i < nr/(int)sizeof(TUPLE); i++) {
	for (int j = 0; j < ns/(int)sizeof(TUPLE); j++) {
	  if (bufR[i].val == bufS[j].val) {
	    result.rkey = bufR[i].key;
	    result.rval = bufR[i].val;
	    result.skey = bufS[j].key;
	    result.sval = bufS[j].val;
	    resultVal += result.rval;
	    // printf should be comment out when measuring performance
	    //printf("%d:%d\t%d:%d\n", result.rkey, result.rval, result.skey, result.sval);
	  }
	}
      }
    }
  }
  gettimeofday(&end, NULL);
  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);

  return 0;
}
