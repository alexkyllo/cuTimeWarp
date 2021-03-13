CC = g++
CFLAGS = -std=c++11 -Wall -Wextra # --pedantic-errors \
#-g -fsanitize=address -fsanitize=leak -fsanitize=undefined -fno-sanitize-recover
#LDFLAGS = -L./inc/lbfgsb-gpu/build/culbfgsb/ -lcuLBFGSB -lblas
LDFLAGS = -lblas
NVCC = nvcc
NVCC_FLAGS = -g -G -maxrregcount=64 -Xcompiler "$(CFLAGS)"
CU_LDFLAGS = -lcublas
.PHONY = default build clean test fmt report
FIGS = img/cost_dependencies.png img/sakoe_chiba.png

# list the CUDA kernel object files
CU_OBJ = obj/euclid_dist.o obj/helper_functions.o obj/soft_dtw.o \
obj/soft_dtw_naive.o \
obj/soft_dtw_naive_multi.o \
obj/soft_dtw_stencil.o \
obj/soft_dtw_tiled.o \

$(shell mkdir -p bin/ obj/)

default: help

## Format the code with clang-format
fmt:
	cd src && clang-format -i *.cpp *.hpp *.cu *.cuh *.hcu

## Build binaries
build: bin/dtw_cpu bin/soft_dtw_perf bin/soft_dtw_perf_multi

bin/dtw_cpu: src/dtw_cpu.cpp
	$(CC) $(CFLAGS) $^ -o $@

bin/timewarp: src/timewarp.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

obj/test.o: test/test.cpp test/catch.h
	$(CC) -std=c++11 test/test.cpp -c -o $@

obj/soft_dtw_perf_main.o: src/soft_dtw_perf_main.cpp
	$(CC) -I$(CUDA_HOME)/include $(CFLAGS) -c $< -o $@

obj/soft_dtw_perf_multi.o: src/soft_dtw_perf_multi.cpp
	$(CC) -I$(CUDA_HOME)/include $(CFLAGS) -c $< -o $@

obj/soft_dtw.o: src/soft_dtw.cu
	$(NVCC) -dc $< -o $@

obj/%.o: src/kernels/%.cu
	$(NVCC) -dc $< -o $@

## Run experiments
run: bin/soft_dtw_perf
	./$< data/ECG200/ECG200_TRAIN.txt output/ECG200/PERFORMANCE.csv

## Run multi-distance experiments
run_multi: bin/soft_dtw_perf_multi
	./$< data/ECG200/ECG200_TRAIN.txt

## Build and run unit tests
test: test_softdtw_cpu test_softdtw_cuda

test_softdtw_cpu: bin/test_soft_dtw_cpu
	./$<

test_softdtw_cuda: bin/test_soft_dtw_cuda
	./$<

bin/test_soft_dtw_cpu: test/test_soft_dtw_cpu.cpp obj/test.o src/soft_dtw_cpu.hpp
	$(CC) $(CFLAGS) $< obj/test.o -o $@ $(LDFLAGS)

bin/test_soft_dtw_cuda: test/test_soft_dtw_cuda.cpp obj/test.o $(CU_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(CU_LDFLAGS)

bin/lbfgs: src/lbfgs_main.cpp src/soft_dtw_cost.hpp
	$(CC) -I./inc/Eigen -I./inc/ $(CFLAGS) src/lbfgs_main.cpp -o bin/lbfgs -lblas

bin/soft_dtw_perf: obj/soft_dtw_perf_main.o $(CU_OBJ)
	$(NVCC) $^ -o $@ $(CU_LDFLAGS)

bin/soft_dtw_perf_multi: obj/soft_dtw_perf_multi.o $(CU_OBJ)
	$(NVCC) $^ -o $@ $(CU_LDFLAGS)

## Compile the PDF report
report: cuTimeWarp.pdf

cuTimeWarp.pdf: cuTimeWarp.tex cuTimeWarp.bib $(FIGS)
	latexmk -pdf

## Delete binaries
clean:
	rm -rf bin/ obj/

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
#	* save line in hold space
#	* purge line
#	* Loop:
#		* append newline + line to hold space
#		* go to next line
#		* if line starts with doc comment, strip comment character off and loop
#	* remove target prerequisites
#	* append hold space (+ newline) to line
#	* replace newline plus comments by `---`
#	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
