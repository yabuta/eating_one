#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/time.h>
#include "debug.h"

#define SZ_PAGE 4096
#define NB_BUF  (SZ_PAGE * 16 / sizeof(TUPLE))
#define NB_BKTENT 4 // the number of partitions

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


void
printDiff(struct timeval begin, struct timeval end)
{
  long diff;
  
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

// create index for S
void
createPart(const char *inFile)
{
  int fd;
  int *bfdAry;
  char partFile[BUFSIZ];
  TUPLE buf[NB_BUF];
  
  fd = open(inFile, O_RDONLY); if (fd == -1) ERR;
  bfdAry = (int *)calloc(NB_BKTENT, sizeof(int));
  for (int i = 0; i < NB_BKTENT; i++) {
    bzero(partFile, sizeof(partFile));
    sprintf(partFile, "hash-part-%s-%d", inFile, i); // part == partition
    bfdAry[i] = open(partFile, O_CREAT|O_WRONLY|O_TRUNC, 0644);
    if (bfdAry[i] == -1) ERR;
  }

  while (1) {
    int n = read(fd, buf, sizeof(buf)); if (n == -1) ERR; else if (!n) break;
    for (unsigned int i = 0; i < n/sizeof(TUPLE); i++) {
      int idx = buf[i].val % NB_BKTENT;
      write(bfdAry[idx], &buf[i], sizeof(TUPLE));
    }
  }

  for (int i = 0; i < NB_BKTENT; i++) close(bfdAry[i]);
  close(fd);
}

int
openPart(const char *partFile, int id)
{
  int fd;
  char buf[BUFSIZ];

  bzero(buf, sizeof(buf));
  sprintf(buf, "hash-part-%s-%d", partFile, id);
  fd = open(buf, O_RDONLY);
  if (fd == -1) ERR;

  return fd;
}

int 
main(void)
{
  int rfd;
  int sfd;
  TUPLE bufR[NB_BUF];
  TUPLE bufS[NB_BUF];
  RESULT result;
  int resultVal = 0;
  struct timeval begin, end;

  // Hash construction phase
  createPart("R");
  createPart("S");

  // Matching phase
  gettimeofday(&begin, NULL);
  for (int i = 0; i < NB_BKTENT; i++) {
    rfd = openPart("R", i);
    sfd = openPart("S", i);

    while (1) {
      int nr = read(rfd, bufR, NB_BUF * sizeof(TUPLE)); if (nr == -1) ERR; else if (nr == 0) break;
      if (nr == 0) break;

      while (1) {
        int ns = read(sfd, bufS, NB_BUF * sizeof(TUPLE)); if (ns == -1) ERR; else if (ns == 0) break;
        if (ns == 0) break;
        
        for (unsigned int j = 0; j < nr/sizeof(TUPLE); j++) {
          for (unsigned int k = 0; k < ns/sizeof(TUPLE); k++) {
            if (bufR[j].val == bufS[k].val) {
              result.rkey = bufR[j].key;
              result.rval = bufR[j].val;
              result.skey = bufS[k].key;
              result.sval = bufS[k].key;
              resultVal += result.rval;
            }
          }
        }
      }
    }
    close(rfd);
    close(sfd);
  }
  gettimeofday(&end, NULL);

  printDiff(begin, end);
  printf("resultVal: %d\n", resultVal);

  return 0;
}


