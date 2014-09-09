#include "thread-hash-bnlj.h"
pthread_mutex_t Lk;

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

void
init(void)
{
  BUFBLK *p;

  if (pthread_mutex_init(&Lk, NULL)) ERR;
  bzero(&HBufBlk, sizeof(BUFBLK));
  p = &HBufBlk;
  for (int i = 0; i < NB_BUFBLK; i++) {
    if (!(p->nxt = (BUFBLK *)calloc(1, sizeof(BUFBLK)))) ERR; p = p->nxt;
    p->doneFileReader = false;
    p->id = i;
  }
}

void *
executor(void *a)
{
  int rfd;
  int nr;
  TUPLE bufR[NB_BUFR];
  TUPLE bufS[NB_BUFS];
  RESULT result;
  int resultVal = 0;
  BUCKET bucket[NB_BUCKET];
  BUFBLK *pbb = HBufBlk.nxt;
  struct timeval begin, end;
  long iodiff = 0;
  long joindiff = 0;

  cpu_set_t cpuset; 
  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset); /* processor 0 */

  /* bind process to processor 0 */
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) < 0) {
    perror("pthread_setaffinity_np");
  }

  gettimeofday(&begin, NULL);
  rfd = open("R", O_RDONLY); if (rfd == -1) ERR;
  bzero(bucket, sizeof(BUCKET) * NB_BUCKET);
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

    int ns;
    int cnt = 0;

    struct timeval bio, eio;
    struct timeval bjoin, ejoin;
    while (true) {
      /* Obtain bufS */
      gettimeofday(&bio, NULL);
      while (true) {
	if (pthread_mutex_lock(&Lk)) ERR;
	if (pbb->doneFileReader == true) {

	  cnt = 0;
	  if (pbb->sz == 0) {
	    if (pthread_mutex_unlock(&Lk)) ERR;
	    gettimeofday(&end, NULL);
	    printDiff(begin, end);
	    printf("resultVal: %d\n", resultVal);
	    printf("iodiff: %ld\n", iodiff);
	    printf("joindiff: %ld\n", joindiff);
	    pthread_exit(NULL);
	  }
	  else {
	    ns = pbb->sz;
	    memcpy(bufS, pbb->buf, ns);

	    /* post processing */
	    pbb->doneFileReader = false;
	    if (pthread_mutex_unlock(&Lk)) ERR;
	    pbb = pbb->nxt; if(!pbb) pbb = HBufBlk.nxt;
	    break;
	  }
	}
	else {
	  cnt++;
	  if (pthread_mutex_unlock(&Lk)) ERR;
	  usleep(1);
	}
      } 
      gettimeofday(&eio, NULL);
      iodiff += calcDiff(bio, eio);

      /* join */
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

  return NULL; // just to complier be quiet
}

void *
fileReader(void *a)
{
  BUFBLK *p;
  int fd;

  cpu_set_t cpuset; 
  CPU_ZERO(&cpuset);
  CPU_SET(1, &cpuset); /* processor 0 */

  /* bind process to processor 1 */
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) < 0) {
    perror("pthread_setaffinity_np");
  }

  fd = open("S", O_RDONLY);
  if (fd == -1) ERR;

  p = HBufBlk.nxt;
  while (1) {
    if (pthread_mutex_lock(&Lk)) ERR;
    if (p->doneFileReader == true) {
      if (pthread_mutex_unlock(&Lk)) ERR;
      usleep(1); continue;
    }
    else {
      p->sz = read(fd, p->buf, NB_BUFS * sizeof(TUPLE));
      p->doneFileReader = true;
      if (pthread_mutex_unlock(&Lk)) ERR;
      
      if (p->sz == 0) {// close & break for exit
	close(fd); break;
      } p = p->nxt; if (!p) p = HBufBlk.nxt;
    }
  }

  return NULL;
}

pthread_t
createFileReader(void)
{
  pthread_t thread;

  if (pthread_create(&thread, NULL, fileReader, NULL)) ERR;
  
  return thread;
}

pthread_t
createExecutor(void)
{
  pthread_t thread;

  if (pthread_create(&thread, NULL, executor, NULL)) ERR;
  
  return thread;
}

void
createThreads(void)
{
  pthread_t thFr, thEx;

  thFr = createFileReader();
  sleep(1);
  thEx = createExecutor();
  if (pthread_join(thFr, NULL)) ERR;
  if (pthread_join(thEx, NULL)) ERR;
}

int
main(void)
{
  init();
  createThreads();

  return 0;
}
