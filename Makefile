CC = g++
CFLAGS = -std=c++11 -Wall -Wextra --pedantic-errors \
-g -fsanitize=address -fsanitize=leak -fsanitize=undefined -fno-sanitize-recover
NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
.PHONY = clean default

$(shell mkdir -p bin/)

bin/timewarp_cpu: timewarp.cpp
	$(CC) $(CFLAGS) $^ -o $@

bin/timewarp: timewarp.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -rf bin/
