T1 = InnerExe
TARGET2 = BuildTable
CC = g++
CFLAGS = -c -Wall -g -DDEBUG -O2
LDFLAGS = -Wall -L/usr/local/lib -lm -lpthread -O2

01 = inner-idx-bnlj.o

#
# Rules for make
#
all: $(T1)
	$(CC) -o $(TARGET2) TableBuild.cpp	
$(T1): $(01) 
	$(CC) -o $@ $^ $(LDFLAGS)


.cc.o:
	$(CC) $(CFLAGS) $<


clean:
	rm -f *~ *.o *.exe *.stackdump hash-part-*

inner-idx-bnlj.o:tuple.h