# Makefile
TARGET	= memtest
CC	= gcc
NVCC	= nvcc -arch sm_35 -cubin
LIBS	= -lcuda -lstdc++
CFLAGS	= -L/usr/local/cuda/lib64 -I /usr/local/cuda/include

all:
	$(NVCC) -o memtest.cubin memtest.cu
	#$(NVCC) -o join_gpu.cubin join_gpuArg.cu
	gcc -o $(TARGET) $(CFLAGS) $(LIBS) memtest.cc

clean:
	rm -f $(TARGET) *.cubin ./*~
