# Makefile
TARGET	= HashExe
CC	= gcc -g 
NVCC	= nvcc -g -G -arch sm_35 -cubin
LIBS	= -lcuda -lstdc++
CFLAGS	= -L/usr/local/cuda/lib64 -I/usr/local/cuda/include

all: hj.cc join_gpu.cu count.cu count_partitioning.cu partitioning.cu
	$(NVCC) -o join_gpu.cubin join_gpu.cu
	$(NVCC) -o count.cubin count.cu
	$(NVCC) -o count_partitioning.cubin count_partitioning.cu
	$(NVCC) -o partitioning.cubin partitioning.cu
	gcc $(CFLAGS) -o $(TARGET) hj.cc $(LIBS)

clean:
	rm -f $(TARGET) *.cubin ./*~
