T1 = Parallel-hashindex

CC = g++
CFLAGS = -c -Wall -g -DDEBUG -O2
LDFLAGS = -Wall -L/usr/local/lib -lm -lpthread -O2

O1 = thread-hash-bnlj.o

#
# Rules for make
#
all: $(T1)

$(T1): $(O1) tuple.h
	$(CC) -o $@ $^ $(LDFLAGS)

.cc.o:
	$(CC) $(CFLAGS) $<

$(O1):tuple.h

clean:
	rm -f *~ *.o $(T1) *.stackdump hash-part-*
