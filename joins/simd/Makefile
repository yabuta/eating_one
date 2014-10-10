T1 = simd.exe
T2 = join-simd-nlj.exe
T3 = join-nlj.exe

CC = g++
CFLAGS = -c -Wall -O3 -mavx
LDFLAGS = -Wall -L/usr/local/lib -lm -lpthread

O1 = simd.o
O2 = join-simd-nlj.o
O3 = join-nlj.o

#
# Rules for make
#
all: $(T1) $(T2) $(T3)

$(T1): $(O1) 
	$(CC) -o $@ $^ $(LDFLAGS)
$(T2): $(O2) 
	$(CC) -o $@ $^ $(LDFLAGS)
$(T3): $(O3) 
	$(CC) -o $@ $^ $(LDFLAGS)

.cc.o:
	$(CC) $(CFLAGS) $<

clean:
	rm -f *~ *.o *.exe *.stackdump hash-part-*
