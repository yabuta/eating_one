T1 = Hjexe

CC = g++
CFLAGS = -c -Wall -g -DDEBUG -O2
LDFLAGS = -Wall -L/usr/local/lib -lm -lpthread -O2

O1 = hj.o

#
# Rules for make
#
all:$(T1) tuple.h


$(T1): $(O1) tuple.h
	$(CC) -o $@ $^ $(LDFLAGS)

.cc.o:
	$(CC) $(CFLAGS) $<

clean:
	rm -f *~ *.o *.exe *.stackdump hash-part-*
