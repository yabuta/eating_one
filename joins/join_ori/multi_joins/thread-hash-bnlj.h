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
#define NB_BUFR (SZ_PAGE * 2 / sizeof(TUPLE))
#define NB_BUFS (SZ_PAGE * 16 / sizeof(TUPLE))
#define NB_BUFBLK 1024

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

typedef struct _BUFBLK { // Buffer Block
  int id;
  TUPLE buf[NB_BUFS];
  bool doneFileReader;
  int sz; // how many bytes are read ?
  struct _BUFBLK *nxt;
} BUFBLK;

typedef struct _HASHOBJ {
  TUPLE tuple;
  struct _HASHOBJ *nxt;
} HASHOBJ;

typedef struct _BUCKET {
  HASHOBJ head;
  HASHOBJ *tail;
} BUCKET;

#define NB_BUCKET NB_BUFR

BUFBLK HBufBlk;
bool FinFileReader;
