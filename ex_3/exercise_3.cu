#define NUM_ITERATIONS 10000
#define ABS(a) ((a) < 0 ? -(a) : (a))
#define DT 1

// Constants
int NUM_PARTICLES;
int BLOCK_SIZE;

// Gravity field
float3 field = (float3) {0.f, 0.f, 9.8f};

// Structure for the particles
typedef struct {
  float3 position;
  float3 velocity;
} Particle;


__device__ void updatePosition(Particle *particle) {
  particle->position.x = particle.position.x + particle.velocity.x * DT;
  particle->position.y = particle.position.y + particle.velocity.y * DT;
  particle->position.z = particle.position.z + particle.velocity.z * DT;
}

__host__ void updatePositionHost(Particle *particle) {
  particle->position.x = particle.position.x + particle.velocity.x * DT;
  particle->position.y = particle.position.y + particle.velocity.y * DT;
  particle->position.z = particle.position.z + particle.velocity.z * DT;
}

__device__ void updateVelocity(Particle *particle) {
  particle->velocity.x = particle.velocity.x + field.x * DT;
  particle->velocity.y = particle.velocity.y + field.y * DT;
  particle->velocity.z = particle.velocity.z + field.z * DT;

}

__host__ void updateVelocityHost(Particle *particle) {
  particle->velocity.x = particle.velocity.x + field.x * DT;
  particle->velocity.y = particle.velocity.y + field.y * DT;
  particle->velocity.z = particle.velocity.z + field.z * DT;

}

__global__ void simulateParticlesKernel(Particle *particles,
    int num_particles, int num_iterations) {

	// Unique ID of the current thread to determine what work to compute
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	// This thread has no work to do, exit
	if (threadId > n) return;

  // Get the right particle
  Particle *particle = particles + threadId;

  for (int i = 0; i < num_iterations; ++i) {
    // Update velocity first
    updateVelocity(particle);

    // Update position
    updatePosition(particle);
  }
}


__host__ void simulateParticlesHost(Particle *particles,
    int num_particles, int num_iterations) {
  
  for (Particle *particle = particles;
      particle < particles + num_particles;
      particles++) {
    for (int i = 0; i < num_iterations; ++i) {
      // Update velocity first
      updateVelocityHost(particle);

      // Update position
      updatePositionHost(particle);
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
void compareSimulationResults(float *deviceOut, float *hostOut, int n) {
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
  if (argc != 3) {
    exit(-1);
  } else {
    NUM_PARTICLES = atoi(argv[1]);
    BLOCK_SIZE = atoi(argv[2]);
  }

	// Allocate memory on the host
	Particle *hostParitcles = (Particles *) malloc(NUM_PARTICLES * sizeof(Particle));

	// Allocate memory on the device
	Particle *devParticles;
	cudaMalloc(&devParticles, NUM_PARTICLES * sizeof(Particle));

	// Fill hostParitcles arrays with random floats
	populateParticleArray(hostParitcles, NUM_PARTICLES);

	// Copy hostParitcles onto the GPU
	cudaMemcpy(devParticles, hostParitcles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

	printf("Simulating particles on the CPU... ");
	double startTime = cpuSecond();
  simulateParticlesHost(hostParticles, NUM_PARTICLES, NUM_ITERATIONS);
	printf("%f seconds\n", cpuSecond() - startTime);

	printf("Simulating particles on the GPU... ");
	startTime = cpuSecond();
	// Round-up to the nearest multiple of BLOCK_SIZE that can hold at least NUM_PARTICLES
	// threads
	simulateParticlesKernel <<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>> (devParticles, NUM_PARTICLES, NUM_ITERATIONS);
	
	// Wait until all the threads on the GPU have finished before continuing!!!
	cudaDeviceSynchronize();
	printf("%f seconds\n", cpuSecond() - startTime);

	// Copy the result of the simulation on the device back to
  // the host into hostParitcles
  Particles *particlesFromGPU = malloc(NUM_PARTICLES * sizeof(Particle));
	cudaMemcpy(particlesFromGPU, devParticles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

	// Compare the results of simulation on device and host
	compareSimulationResults(hostParitcles, particlesFromGPU, NUM_PARTICLES);

	// Free the allocated memory!!!
	free(hostParitcles);
	free(particlesFromGPU);
	cudaFree(devParticles);

	return 0;

}
