CC = g++
CFLAGS = -std=c++11 -Wall -Wextra --pedantic-errors
NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
.PHONY = clean default

$(shell mkdir -p bin/)

bin/timewarp: timewarp.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -rf bin/
