# Makefile
TARGET	= ExeNestLoop
TARGET2 = BuildTable
CC	= gcc
NVCC	= nvcc -arch sm_35 -cubin
LIBS	= -lcuda -lstdc++
CFLAGS	= -L/usr/local/cuda/lib64 -I /usr/local/cuda/include

all:
#	gcc -g -o $(TARGET) $(CFLAGS) $(LIBS) nestLoopJoin.cc
	g++ -o $(TARGET) nestLoopJoin.cc
	g++ -o $(TARGET2) TableBuild.cpp
clean:
	rm -f $(TARGET) $(TARGET2) *.cubin ./*~
