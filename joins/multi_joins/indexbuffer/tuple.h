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

#define THREAD_NUM 8

#define JT_SIZE 100000000
#define SELECTIVITY 1000000000
#define MATCH_RATE 0.01

#define LT_BUF 262144

#define NB_BUCKET 134217728

#define BUFF_SIZE 1024


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


typedef struct _BUCKET {
    int val;
    int adr;

} BUCKET;


typedef struct _LTBUFDATA { // Buffer Block
  TUPLE *startPos;
  int size;

} LTBUFDATA;

