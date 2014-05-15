#Makefile
TARGET = main
CC = gcc -g
LIBS = -lcuda -lstdc++
CFLAGS = -L/usr/local/cuda/lib64 -I/usr/local/cuda/include

all: main.cpp
	gcc $(CFLAGS) -o $(TARGET) main.cpp $(LIBS)

clean:
	rm -f $(TARGET) ./*~ *.o