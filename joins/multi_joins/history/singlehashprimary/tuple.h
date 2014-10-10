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
#define SELECTIVITY 100000000
#define MATCH_RATE 0.5

#define PARTITION 1048576

#define LT_BUF 16384

/*

4194304*4194304
8388608*8388608
16777216*16777216  1048576
33554432*33554432  262144
67108864*67108864
134217728*134217728


*/


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




typedef struct _LTBUFDATA { // Buffer Block
    TUPLE *startPos;
    int size;

} LTBUFDATA;

