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
#define MATCH_RATE 0.1

#define PARTITION 1048576

#define BUFF_SIZE 1024



/*PARTITION number

4194304*4194304    32768
8388608*8388608    65536
16777216*16777216  131072
33554432*33554432  262144
67108864*67108864  524288
134217728*134217728 1048576




1048576*67108864    131072
4194304*67108864    262144
8388608*67108864    262144
16777216*67108864   524288



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




typedef struct _RTBUFDATA { // Buffer Block
    TUPLE *lstartPos;
    int lsize;
    TUPLE *rstartPos;
    int rsize;

} RTBUFDATA;

