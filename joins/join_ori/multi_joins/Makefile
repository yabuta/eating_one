T1 = outer-idx-bnlj.exe
T2 = inner-idx-bnlj.exe
T3 = hj.exe
T4 = bnlj.exe
T5 = thread-hash-bnlj.exe

CC = g++
CFLAGS = -c -Wall -g -DDEBUG -O2
LDFLAGS = -Wall -L/usr/local/lib -lm -lpthread -O2

O1 = outer-idx-bnlj.o
O2 = inner-idx-bnlj.o
O3 = hj.o
O4 = bnlj.o
O5 = thread-hash-bnlj.o

#
# Rules for make
#
all: $(T1) $(T2) $(T3) $(T4) $(T5)

$(T1): $(O1) 
	$(CC) -o $@ $^ $(LDFLAGS)
$(T2): $(O2) 
	$(CC) -o $@ $^ $(LDFLAGS)
$(T3): $(O3) 
	$(CC) -o $@ $^ $(LDFLAGS)
$(T4): $(O4) 
	$(CC) -o $@ $^ $(LDFLAGS)
$(T5): $(O5) 
	$(CC) -o $@ $^ $(LDFLAGS)

.cc.o:
	$(CC) $(CFLAGS) $<

clean:
	rm -f *~ *.o *.exe *.stackdump hash-part-*
