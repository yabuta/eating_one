# Makefile
TARGET	= NestLoopExe
CC	= gcc
NVCC	= nvcc -arch sm_35 -cubin
LIBS	= -lcuda -lstdc++
CFLAGS	= -L/usr/local/cuda/lib64 -I /usr/local/cuda/include

all:
	$(NVCC) -o join_gpu.cubin join_gpuArg.cu
	gcc -o $(TARGET) $(CFLAGS) $(LIBS) nestLoopJoinArg.cc

clean:
	rm -f $(TARGET) *.cubin ./*~
