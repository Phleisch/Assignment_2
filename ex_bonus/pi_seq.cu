#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Necessary for random numbers in CUDA
#include <curand_kernel.h>
#include <curand.h>

#define SEED     921
#define NUM_ITER 1000000000
#define TPB 256				// Threads PER block
#define NUM_THREADS 5000	// Total number of threads to execute

/**
 * Function which, for each instance of the kernel, generates a random point
 * and calculates whether or not it is within a circle.
 *
 * @param counts		Array for each thread to store the total number of
 *						randomly generate points that were within a circle
 * @param numIter		Number of iterations / points each thread should make
 * @param curandState	Array for each thread to store its own curandState
 *						structure
 */
__global__ void estimatePiKernel(unsigned int *counts, unsigned int numIter,
									curandState *randState) {
	double x, y, distance;

	// Unique ID of the current thread to determine what work to compute
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	// This thread has no work to do, exit
	if (threadId > /*TODO*/) return;
	
	// Used threadId as a seed of randomness so that every thread is generating
	// different random values
	int seed = threadId;
	curand_init(threadId, threadId, 0, &states[threadId]);

	for (int iter = 0; iter < numIter; iter++) {
		// Generate random x, y coordinates from 0.0 (exclusive) to 1.0
		// (inclusive) for a point
		x = (double) curand_uniform(&states[threadId]);
		y = (double) curand_uniform(&states[threadId]);

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
void estimatePi(int *counts) {
	int totalCount = 0;

	// accumulate the counts of coins in the circle into totalCount
	for (int index = 0; index < (NUM_ITER / NUM_THREADS); index++) {
		totalCount += counts[index];
	}

	// Calculate pi according to the formula P(coin in circle) * 4 where
	// P(coin in circle) is equivalents to (coins in circle) / (total coins)
	double piEstimation = ((double) totalCount / (double) NUM_ITER) * 4.0;
	printf("The result is %f\n", piEstimation);
}

int main() {
	
	// Allocate space for curandState for each thread
	curandState *randState;
	cudaMalloc(&randState, NUM_ITER * sizeof(devRandom));

	// Allocate space to keep track of counts of points generated in the circle
	unsigned int *deviceCounts;
	cudaMalloc(&deviceCounts, (NUM_ITER / NUM_THREADS) * sizeof(unsigned int));

	// Allocate space to copy the GPU result back to the CPU
	unsigned int *hostCounts = (unsigned int*) malloc(
		(NUM_ITER / NUM_THREADS) * sizeof(unsigned int))

	// Set all of the memory to 0
	cudaMemset(counts, 0, (NUM_ITER / NUM_THREADS) * sizeof(unsigned int));

	// Launch the kernel
	generateAndCountPointsKernel <<<(NUM_THREADS + TPB - 1) / TPB, TPB>>> (
		deviceCounts, NUM_ITER / NUM_THREADS, randState);
	
	// Watch for the kernel to finish
	cudaDeviceSynchronize();

	// Copy GPU counts to the CPU
	cudaMemcpy(
		hostCounts, deviceCounts,
		(NUM_ITER / NUM_THREADS) * sizeof(unsigned int), cudaMemcpyDeviceToHost
	);

	// Print pi estimation
	estimatePi(hostCounts);

	return 0;
}
