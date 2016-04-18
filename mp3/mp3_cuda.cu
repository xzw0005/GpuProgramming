// Heat Transfer Simulation
// MP3, Spring 2016, GPU Programming @ Auburn University
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>

#define N 131072              // Number of points in the rod
#define INITIAL_LEFT  1000.0  // Initial temperature at left end
#define INITIAL_RIGHT 0.0     // Initial temperature at right end
#define ALPHA 0.5             // Constant
#define MAX_TIMESTEPS 10000   // Maximum number of time steps
#define THREADS_PER_BLOCK 256

static void check_result(double *result);

__global__ static void calcHeat(double *ot, double *nt) {
    //for (int i = 1; i < N-1; i++) {
    //    new_t[i] = old_t[i] + ALPHA*(old_t[i-1] + old_t[i+1] - 2*old_t[i]);
    //}
    //new_t[N-1] = old_t[N-1];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > 0 && i < N - 1) {
        nt[i] = ot[i] + ALPHA * (ot[i - 1] + ot[i + 1] - 2 * ot[i]);
    }
}
__global__ static void swapTemp(double *ot, double *nt, double *tt) {
    // Swap old and new buffers for next iteration (double-buffering)
    //temp = old_t;
    //old_t = new_t;
    //new_t = temp;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        tt[i] = ot[i];
        ot[i] = nt[i];
        nt[i] = ot[i];
    }
}


int main() {
    double *old_t = (double *)malloc(N * sizeof(double));
    double *new_t = (double *)malloc(N * sizeof(double));
    double *temp = (double *)malloc(N * sizeof(double));

    // Initialize arrays/set initial values
    old_t[0] = INITIAL_LEFT;
    for (int i = 1; i < N-1; i++) {
        old_t[i] = 0.0;
    }
    old_t[N-1] = INITIAL_RIGHT;
	new_t[0] = old_t[0];
	new_t[N-1] = old_t[N-1];
    
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    double start = omp_get_wtime();

    double *d_old, *d_new, *d_temp;
    cudaMalloc((void **)&d_old, N * sizeof(double));
    cudaMalloc((void **)&d_new, N * sizeof(double));
    cudaMalloc((void **)&d_temp, N * sizeof(double));

    cudaMemcpy(old_t, d_old, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(new_t, d_new, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(temp, d_temp, N * sizeof(double), cudaMemcpyHostToDevice);

    // Compute temperatures at each sample point in the rod over time
    int time;
    for (time = 0; time < MAX_TIMESTEPS; time++) {
        calcHeat<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_old, d_new);
        swapTemp<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_old, d_new, d_temp);
    }

    cudaMemcpy(old_t, d_old, N * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_old);
    cudaFree(d_new);

    double stop = omp_get_wtime();

    // Show output (final temperatures)
    printf("Stopped after %d time steps\n", time);
    printf("Simulation took %f seconds\n", stop - start);
    
    check_result(old_t);
    return 0;
}

static void check_result(double *result) {
    char output[1024] = { 0 };
    char *out = output;
    
    // Display some of the computed results
    for (int i = 0; i < 6; i++) {
        out += sprintf(out, "%3.3f ", result[i]);
    }
    out += sprintf(out, "... ");
    for (int i = N-6; i < N; i++) {
        out += sprintf(out, "%3.3f ", result[i]);
    }
    printf("Computed: %s\n", output);

    // Display the expected output
    const char *expected = "1000.000 992.021 984.044 976.067 968.095 960.123 ... 0.000 0.000 0.000 0.000 0.000 0.000 ";
    printf("Expected: %s\n", expected);

    // Exit with a nonzero exit code if the two do not match
    assert(strcmp(output, expected) == 0);
}