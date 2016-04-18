/* Heat Transfer Simulation
 * MP3, Spring 2016, GPU Programming @ Auburn University
 * @author: Xing Wang
 *
 * Compile with:
 * nvcc -O3 -Xcompiler=-fopenmp -o mp3_acc mp3_acc.c
 */
 
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>

#define N 131072              // Number of points in the rod
#define INITIAL_LEFT  1000.0  // Initial temperature at left end
#define INITIAL_RIGHT 0.0     // Initial temperature at right end
#define ALPHA 0.5             // Constant
#define MAX_TIMESTEPS 10000   // Maximum number of time steps

static void check_result(double *result);

int main() {
    double *old_t = (double *)malloc(N * sizeof(double));
    double *new_t = (double *)malloc(N * sizeof(double));
    double *temp;

    acc_init(acc_device_default);

    // Initialize arrays/set initial values
    old_t[0] = INITIAL_LEFT;
    for (int i = 1; i < N-1; i++) {
        old_t[i] = 0.0;
    }
    old_t[N-1] = INITIAL_RIGHT;

    double start = omp_get_wtime();

    // Compute temperatures at each sample point in the rod over time
    int time;
    new_t[0] = old_t[0];
    new_t[N-1] = old_t[N-1];
    #pragma acc data copy(old_t[0:N], new_t[1:N-1])
    for (time = 0; time < MAX_TIMESTEPS; time++) {
        #pragma acc parallel loop
        for (int i = 1; i < N - 1; i++) {
            new_t[i] = old_t[i] + ALPHA*(old_t[i-1] + old_t[i+1] - 2*old_t[i]);
        }
        // Swap old and new buffers for next iteration (double-buffering)
        #pragma acc parallel loop
        for (int i = 1; i < N - 1; i++) {
            old_t[i] = new_t[i];
        }
        //temp = old_t;
        //old_t = new_t;
        //new_t = temp;
    }
    
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