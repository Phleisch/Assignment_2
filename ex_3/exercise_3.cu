#include <stdio.h>			// For use of the printf function
#include <sys/time.h>		// For use of gettimeofday function

#define NUM_ITERATIONS 10000
#define ABS(a) ((a) < 0 ? -(a) : (a))
#define DT 1

int NUM_PARTICLES;	// # of particles to simulate, equivalent to # of threads
int BLOCK_SIZE;		// Threads PER block

// Gravity field
float3 field = (float3) {0.f, 0.f, 9.8f};

// Structure for the particles
typedef struct {
  float3 position;
  float3 velocity;
} Particle;

/**
 * Can use multiple qualifiers to specify where a function will run in order
 * to reuse code that needs to be run on both host and device.
 * Change the position of the given particle based on its velocity using the
 * formula `new_position.coord = old_position.coord + velocity.coord` where
 * coord is x, y and z.
 *
 * @param particle	Particle for which a position update will be performed
 */
__host__ __device__ void updatePosition(Particle *particle) {
  particle->position.x = particle->position.x + particle->velocity.x * DT;
  particle->position.y = particle->position.y + particle->velocity.y * DT;
  particle->position.z = particle->position.z + particle->velocity.z * DT;
}

/**
 * Update the velocity of the given particle according to a field that specifies
 * the rate of change for each dimension of the particle's velocity
 *
 * @param particle	Particle for which a velocity update will be performed
 * @param field		Rate of change for each dimension (x, y, z) of a velocity
 */
__host__ __device__ void updateVelocity(Particle *particle, float3 field) {
  particle->velocity.x = particle->velocity.x + field.x * DT;
  particle->velocity.y = particle->velocity.y + field.y * DT;
  particle->velocity.z = particle->velocity.z + field.z * DT;

}

/**
 * Device implementation for the simulation of moving particles
 *
 * @param particles			List of particles for which to simulate movement
 * @param field				Values specifying the rate of change for a
 *							particle's velocity in each dimension
 * @param num_particles		# of particles, used to determine how many threads
 *							to give work if too many threads are initiated
 * @param num_iterations	# of timesteps a thread should simulate a particle
 */
__global__ void simulateParticlesKernel(Particle *particles, float3 field,
    int num_particles, int num_iterations) {

	// Unique ID of the current thread to determine what work to compute
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	// This thread has no work to do, exit
	if (threadId > num_particles) return;

  // Get the right particle
  Particle *particle = particles + threadId;

  for (int i = 0; i < num_iterations; ++i) {
    // Update velocity first
    updateVelocity(particle, field);

    // Update position
    updatePosition(particle);
  }
}

/**
 * Host implementation for the simulation of moving particles
 *
 * @param particles			List of particles for which to simulate movement
 * @param num_particles		# of particles to simulate
 * @param num_iterations	# of timesteps for which to simulate each particle
 */
__host__ void simulateParticlesHost(Particle *particles,
    int num_particles, int num_iterations) {
  
  for (Particle *particle = particles;
      particle < particles + num_particles;
      particle++) {
    for (int i = 0; i < num_iterations; ++i) {
      // Update velocity first
      updateVelocity(particle, field);

      // Update position
      updatePosition(particle);
    }
  }
}

/**
 * Fill the given array with n random floats.
 *
 * @param array	Array to populate with floats.
 * @param n		Number of floats to populate the array with.
 */
void populateParticleArray(Particle *particles, int n) {
  Particle particle;

	for (int index = 0; index < n; index++) {
		// Generate random particles
    particle.position.x = 10.0 * ((float) rand() / (float) RAND_MAX);
    particle.position.y = 10.0 * ((float) rand() / (float) RAND_MAX);
    particle.position.z = 10.0 * ((float) rand() / (float) RAND_MAX);
    particle.velocity.x = 1.0 * ((float) rand() / (float) RAND_MAX);
    particle.velocity.y = 1.0 * ((float) rand() / (float) RAND_MAX);
    particle.velocity.z = 1.0 * ((float) rand() / (float) RAND_MAX);

		particles[index] = particle;
	}
}

/**
 * Compare the simulation results of the device and host implementations.
 *
 * @param deviceOut	Outcome of simulation from the device implementation
 * @param hostOut	Outcome of simulation from the host implementation
 * @param n			The size of deviceOut and hostOut
 */
void compareSimulationResults(Particle *deviceOut, Particle *hostOut, int n) {
	bool resultsAreEqual = true;
	printf("Comparing the output for each implementation... ");

	for (int index = 0; index < n; index++) {
		float cumDiff = 0;
    cumDiff += ABS(deviceOut[index].position.x - hostOut[index].position.x);
    cumDiff += ABS(deviceOut[index].position.y - hostOut[index].position.y);
    cumDiff += ABS(deviceOut[index].position.z - hostOut[index].position.z);
    cumDiff += ABS(deviceOut[index].velocity.x - hostOut[index].velocity.x);
    cumDiff += ABS(deviceOut[index].velocity.y - hostOut[index].velocity.y);
    cumDiff += ABS(deviceOut[index].velocity.z - hostOut[index].velocity.z);

		// Difference is larger than rounding-error tolerance of .001, means
		// the outcomes are too different
		if (cumDiff > .001 || cumDiff < -.001) {
			resultsAreEqual = false;
			break;
		}
	}

	// The outcomes of simulation for the device and host implementations are equal
	if (resultsAreEqual) {
		printf("Correct!\n");
	} else {
		printf("INCORRECT!!!\n");
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

// Entry point into the program, run each implementation of simulation and compare
// the results
int main(int argc, char **argv) {
  char *file_path;
  FILE *out_file = 0;
  if (argc != 3 && argc != 4) {
    printf("Usage: %s <num_particles> <block_size> [output_file]\n", argv[0]);
    exit(-1);
  } else {
    NUM_PARTICLES = atoi(argv[1]);
    BLOCK_SIZE = atoi(argv[2]);
    if (argc == 4)
      file_path = argv[3];
  }

  if (file_path)
    out_file = fopen(file_path, "a");

	// Allocate memory on the host
	Particle *hostParitcles = (Particle *) malloc(NUM_PARTICLES * sizeof(Particle));

	// Allocate memory on the device
	Particle *devParticles;
	cudaMalloc(&devParticles, NUM_PARTICLES * sizeof(Particle));

	// Fill hostParitcles arrays with random floats
	populateParticleArray(hostParitcles, NUM_PARTICLES);

	// Copy hostParitcles onto the GPU
	cudaMemcpy(devParticles, hostParitcles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

  double startTime = cpuSecond();

  if (NUM_PARTICLES < 100001) {
    printf("Simulating particles on the CPU...\n");
    simulateParticlesHost(hostParitcles, NUM_PARTICLES, NUM_ITERATIONS);
    if (out_file)
      fprintf(out_file, "cpu,%d,%d,%d,%f\n", NUM_PARTICLES, BLOCK_SIZE, NUM_ITERATIONS,
          cpuSecond() - startTime);
    printf("%f seconds\n", cpuSecond() - startTime);
  }

	printf("Simulating particles on the GPU... ");
	startTime = cpuSecond();
	// Round-up to the nearest multiple of BLOCK_SIZE that can hold at least NUM_PARTICLES
	// threads
	simulateParticlesKernel <<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>> (devParticles, field, NUM_PARTICLES, NUM_ITERATIONS);
	
	// Wait until all the threads on the GPU have finished before continuing!!!
	cudaDeviceSynchronize();
  if (out_file)
    fprintf(out_file, "gpu,%d,%d,%d,%f\n", NUM_PARTICLES, BLOCK_SIZE, NUM_ITERATIONS,
        cpuSecond() - startTime);
	printf("%f seconds\n", cpuSecond() - startTime);

	// Copy the result of the simulation on the device back to
  // the host into hostParitcles
  Particle *particlesFromGPU = (Particle *) malloc(NUM_PARTICLES * sizeof(Particle));
	cudaMemcpy(particlesFromGPU, devParticles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

	// Compare the results of simulation on device and host
	compareSimulationResults(hostParitcles, particlesFromGPU, NUM_PARTICLES);

	// Free the allocated memory!!!
	free(hostParitcles);
	free(particlesFromGPU);
	cudaFree(devParticles);
  if (file_path)
    fclose(out_file);

	return 0;

}
