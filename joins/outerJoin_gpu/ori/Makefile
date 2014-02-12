T1 = outer-idx-bnlj.exe

CC = g++
CFLAGS = -c -Wall -g -DDEBUG -O2
LDFLAGS = -Wall -L/usr/local/lib -lm -lpthread -O2

O1 = outer-idx-bnlj.o

#
# Rules for make
#
all: $(T1)

$(T1): $(O1) 
	$(CC) -o $@ $^ $(LDFLAGS)

.cc.o:
	$(CC) $(CFLAGS) $<

clean:
	rm -f *~ *.o *.exe *.stackdump hash-part-*
