all: exercise_3.out

exercise_3.out: exercise_3.cu
	nvcc -arch=sm_50 exercise_3.cu -o exercise_3.out

test: exercise_3.out
	bash run.sh

.PHONY: all test
