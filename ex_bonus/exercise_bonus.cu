#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// Necessary for random numbers in CUDA
#include <curand_kernel.h>
#include <curand.h>

#define NUM_ITER 1000000000
#define TPB 128				// Threads PER block
#define NUM_THREADS 10000	// Total number of threads to execute

/**
 * Function which, for each instance of the kernel, generates a random point
 * and calculates whether or not it is within a circle.
 *
 * @param counts		Array for each thread to store the total number of
 *						randomly generate points that were within a circle
 * @param numIter		Number of iterations / points each thread should make
 * @param numThreads	Number of threads that should be doing work
 * @param curandState	Array for each thread to store its own curandState
 *						structure
 */
__global__ void estimatePiKernel(unsigned int *counts, unsigned int numIter,
					unsigned int numThreads,
					curandState *randState) {
	double x, y, distance;

	// Unique ID of the current thread to determine what work to compute
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	// This thread has no work to do, exit
	if (threadId > numThreads) return;
	
	// Used threadId as a seed of randomness so that every thread is generating
	// different random values
	int seed = threadId;
	curand_init(threadId, seed, 0, &randState[threadId]);

	for (int iter = 0; iter < numIter; iter++) {
		// Generate random x, y coordinates from 0.0 (exclusive) to 1.0
		// (inclusive) for a point
		x = (double) curand_uniform(&randState[threadId]);
		y = (double) curand_uniform(&randState[threadId]);
		
		// Distance from the origin of the circle 
		distance = sqrt((x * x) + (y * y));

		// If the distance from the origin of the circle is less than or equal
		// to 1, that means that the randomly generated point is inside the
		// circle because the circle has a radius of 1. Increment number of
		// points randomly generated within the circle
		if (distance <= 1.0) counts[threadId]++;
	}
}

/**
 * Tally up the counts in an array indicating the number of randomly generated
 * points that were inside a circle and estimate pi.
 *
 * @param counts	Array of counts of points generate inside a circle
 */
void estimatePi(unsigned int *counts) {
	unsigned int totalCount = 0;

	// accumulate the counts of coins in the circle into totalCount
	for (int index = 0; index < NUM_THREADS; index++) {
		totalCount += counts[index];
	}
	printf("total count: %d\n", totalCount);

	// Calculate pi according to the formula P(coin in circle) * 4 where
	// P(coin in circle) is equivalents to (coins in circle) / (total coins)
	double piEstimation = ((double) totalCount / (double) NUM_ITER) * 4.0;
	printf("The result is %f\n", piEstimation);
}

/**
 * Return a timestamp with double percision.
 */
double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main() {
	
	// Allocate space for curandState for each thread
	curandState *randState;
	cudaMalloc(&randState, NUM_THREADS * sizeof(curandState));

	// Allocate space to keep track of counts of points generated in the circle
	unsigned int *deviceCounts;
	cudaMalloc(&deviceCounts, NUM_THREADS * sizeof(unsigned int));

	// Allocate space to copy the GPU result back to the CPU
	unsigned int *hostCounts = (unsigned int*) malloc(
		NUM_THREADS * sizeof(unsigned int));

	// Set all of the memory to 0
	cudaMemset(deviceCounts, 0, NUM_THREADS * sizeof(unsigned int));


	double startTime = cpuSecond();
	// Launch the kernel
	estimatePiKernel <<<(NUM_THREADS + TPB - 1) / TPB, TPB>>> (
		deviceCounts, NUM_ITER / NUM_THREADS, NUM_THREADS, randState);
	
	// Watch for the kernel to finish
	cudaDeviceSynchronize();
	printf("Total time: %f\n", cpuSecond() - startTime);

	// Copy GPU counts to the CPU
	cudaMemcpy(
		hostCounts, deviceCounts,
		NUM_THREADS * sizeof(unsigned int), cudaMemcpyDeviceToHost
	);

	// Print pi estimation
	estimatePi(hostCounts);

	return 0;
}
