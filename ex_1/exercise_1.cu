#include <stdio.h>	// For use of the printf function
#define N 256		// Number of threads to use
#define TPB 256		// Threads PER block

/**
 * Function launched from the CPU and run on the GPU that will display a message
 * of the format `Hello World! My threadId is x` where x is the the threadId of
 * the thread found by thread indexing.
 */
__global__ void helloWorldKernel() {

	// Calculate a unique ID for the currently running thread by using its
	// index within a block plus an offset of number of threads before the
	// current block (blockIdx.x * blockDim.x)
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Print a message of format 'Hello World! My threadId is x' where x is
	// thread_id
	printf("Hello World! My threadId is %d\n", threadId);
}

// Entry point into the program, initiate kernel launch
int main() {

	// Run the helloWorldKernel function on device using N / TPB thread
	// blocks of TPB threads PER block
	helloWorldKernel <<<(N / TPB), TPB>>> ();

	// CRUCIAL! Make execution of the kernel synchronous so that the CPU waits
	// for all threads in the grid to complete before finishing the program.
	// Without this you will not see the result of calls to printf in the kernel
	// function due to the default asynchronous nature of kernel launches
	cudaDeviceSynchronize();

	return 0;
}
