# Makefile
TARGET	= InnerExe
CC	= gcc
NVCC	= nvcc -arch sm_35 -cubin
LIBS	= -lcuda -lstdc++
CFLAGS	= -L/usr/local/cuda/lib64 -I /usr/local/cuda/include

all: inner-idx-bnlj.cc join_gpuArg.cu
	$(NVCC) -o join_gpu.cubin join_gpu.cu
	$(NVCC) -o count.cubin count.cu
	gcc -o $(TARGET) $(CFLAGS) $(LIBS) inner-idx-bnlj.cc 

clean:
	rm -f $(TARGET) *.cubin ./*~
