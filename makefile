# CC and CFLAGS are varilables
CC = g++
CFLAGS = -c
AR = ar
ARFLAGS = rcv
# -c option ask g++ to compile the source files, but do not link.
# -g option is for debugging version
# -O2 option is for optimized version
DBGFLAGS = -g -D_DEBUG_ON_
OPTFLAGS = -O2


NVCC = nvcc
CUDAFLAGS= -std=c++11

all	: cuda main
	@echo -n ""

# optimized version
main	: tool_opt.o main_opt.o lib
		$(CC) $(OPTFLAGS) tool_opt.o main_opt.o -ltm_usage -Llib -o bin/run
main_opt.o 	   	: src/main.cpp lib/tm_usage.h
			$(CC) $(CFLAGS) $< -Ilib -o $@
tool_opt.o	: src/tool.cpp include/tool.h
			$(CC) $(CFLAGS) $(OPTFLAGS) $< -o $@

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib/libwb/lib
cuda	:cuda.o lib
		$(NVCC) cuda.o -lwb -Llib/libwb/lib -o bin/cuda
cuda.o	:src/mp3.cu lib
		$(NVCC) $(CUDAFLAGS) -c $< -Ilib/libwb -o $@

lib: lib/libtm_usage.a

lib/libtm_usage.a: tm_usage.o
	$(AR) $(ARFLAGS) $@ $<
tm_usage.o: lib/tm_usage.cpp lib/tm_usage.h
	$(CC) $(CFLAGS) $<

# clean all the .o and executable files
clean:
		rm -rf *.o lib/*.a lib/*.o bin/*

