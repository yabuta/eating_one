T1 = Parallel-hash
T2 = BuildTable

CC = g++
CFLAGS = -c -Wall -g -DDEBUG -O2
LDFLAGS = -Wall -L/usr/local/lib -lm -lpthread -O2

O1 = thread-hash-bnlj.o

#
# Rules for make
#
all: $(T1)
	$(CC) -o $(T2) TableBuild.cpp


$(T1): $(O1) tuple.h
	$(CC) -o $@ $^ $(LDFLAGS)

.cc.o:
	$(CC) $(CFLAGS) $<

$(O1):tuple.h

clean:
	rm -f *~ *.o $(T1) $(T2) *.stackdump hash-part-*
