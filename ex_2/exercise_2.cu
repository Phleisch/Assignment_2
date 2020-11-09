#include <stdio.h>			// For use of the printf function
#include <sys/time.h>		// For use of gettimeofday function
#define A 77				// Constant A to be used in SAXPY computations
#define ARRAY_SIZE 10000	// Size of arrays to be used in SAXPY computations
#define TPB 256				// Threads PER block

// SAXPY means "Single-Precision A*X PLUS Y" where A is a constant and X, Y are
// arrays

/**
 * Implementation of SAXPY to be performed on device
 *
 * @param x	Array X of SAXPY
 * @param y	Array Y of SAXPY
 * @param a Constant A of SAXPY
 * @param n Number of elements in each array X and Y
 */
__global__ void saxpyKernel(float *x, float *y, float a, int n) {
	
	// Unique ID of the current thread to determine what work to compute
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	// This thread has no work to do, exit
	if (threadId > n) return;

	// Compute SAXPY on one element of x and y based on the id of this thread
	float saxpyResult = (a * x[threadId]) + y[threadId];

	// Save the result of single element SAXPY
	y[threadId] = saxpyResult;
}

/**
 * Implementation of SAXPY to be performed on host
 *
 * @param x	Array X of SAXPY
 * @param y	Array Y of SAXPY
 * @param a Constant A of SAXPY
 * @param n Number of elements in each array X and Y
 */
__host__ void saxpyHost(float *x, float *y, float a, int n) {

	// Simple for-loop solution for the host version
	for (int index = 0; index < n; index++) {
		y[index] = (a * x[index]) + y[index];
	}
}

/**
 * Compare the SAXPY results of the device and host implementations.
 *
 * @param deviceOut	Outcome of SAXPY from the device implementation
 * @param hostOut	Outcome of SAXPY from the host implementation
 * @param n			The size of deviceOut and hostOut
 */
void compareSaxpyResults(float *deviceOut, float *hostOut, int n) {
	bool resultsAreEqual = true;
	printf("Comparing the output for each implementation... ");

	for (int index = 0; index < n; index++) {
		float diff = deviceOut[index] - hostOut[index];

		// Difference is larger than rounding-error tolerance of .01, means
		// the outcomes are too different
		if (diff > .01 || diff < -.01) {
			resultsAreEqual = false;
		}
	}

	// The outcomes of SAXPY for the device and host implementations are equal
	if (resultsAreEqual) {
		printf("Correct!\n")
	} else {
		printf("INCORRECT!!!\n");
	}
}

/**
 * Fill the given array with n random floats.
 *
 * @param array	Array to populate with floats.
 * @param n		Number of floats to populate the array with.
 */
void populateArrayWithFloats(float *array, int n) {
	for (int index = 0; index < 0; index++) {
		array[index] = (float) rand();
	}
}

/**
 * Return a timestamp with double precision.
 */
double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

// Entry point into the program, run each implementation of SAXPY and compare
// the results
int main() {
	// Allocate memory on the host
	float *hostX = (float*) malloc(ARRAY_SIZE * sizeof(float));
	float *hostY = (float*) malloc(ARRAY_SIZE * sizeof(float));

	// Allocate memory on the device
	float *devX, *devY;
	cudaMalloc(&devX, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&devY, ARRAY_SIZE * sizeof(float));

	// Fill hostX and hostY arrays with random floats
	populateArrayWithFloats(hostX, ARRAY_SIZE);
	populateArrayWithFloats(hostY, ARRAY_SIZE);

	// Copy hostX and hostY onto the GPU
	cudaMemcpy(devX, hostX, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devY, hostX, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	printf("Computing SAXPY on the CPU... ");
	double startTime = cpuSecond();
	saxpyHost(hostX, hostY, A, ARRAY_SIZE);
	printf("%f seconds\n", cpuSecond() - startTime);

	printf("Computing SAXPY on the GPU... ");
	startTime = cpuSecond();
	// Round-up to the nearest multiple of TPB that can hold at least ARRAY_SIZE
	// threads
	saxpyKernel <<<(ARRAY_SIZE + TPB - 1) / TPB, TPB>>> (
		devX, devY, A, ARRAY_SIZE);
	
	// Wait until all the threads on the GPU have finished before continuing!!!
	cudaDeviceSynchronize();
	printf("%f seconds\n", cpuSecond() - startTime);

	// Copy the result of SAXPY on the device back to the host into hostX
	cudaMemcpy(hostX, devY, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	// Compare the results of SAXPY on device and host
	compareSaxpyResults(hostX, hostY, ARRAY_SIZE);

	// Free the allocated memory!!!
	free(hostX);
	free(hostY);
	cudaFree(devX);
	cudaFree(devY);

	return 0;
}
