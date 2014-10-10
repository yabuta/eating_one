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

/*
#define SZ_PAGE 4096
#define NB_BUFR (SZ_PAGE * 2 / sizeof(TUPLE))
#define NB_BUFS (SZ_PAGE * 16 / sizeof(TUPLE))
#define NB_BUFBLK 1024
*/

#define THREAD_NUM 8

#define JT_SIZE 100000000
#define SELECTIVITY 1000000000
#define MATCH_RATE 0.01

#define RT_BUF 8192


#define DISTANCE 1

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




typedef struct _RTBUFDATA { // Buffer Block
  TUPLE *startPos;
  int size;

} RTBUFDATA;

