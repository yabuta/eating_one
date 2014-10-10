T1 = HashExe
T2 = BuildTable

CC = g++
CFLAGS = -c -Wall -g -DDEBUG -O2
LDFLAGS = -Wall -L/usr/local/lib -lm -lpthread -O2

01 = hj.o

#
# Rules for make
#
all: $(T1)
	$(CC) -o $(T2) TableBuild.cpp

$(T1): $(01) 
	$(CC) -o $@ $^ $(LDFLAGS)

.cc.o:
	$(CC) $(CFLAGS) $<

clean:
	rm -f *~ *.o $(T1) $(T2) *.exe *.stackdump hash-part-*

hj.o:tuple.h