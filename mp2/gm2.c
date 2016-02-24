#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>

#define N 8000		/* Matrix width/height */
#define PRINT 10	/* Number of rows/columns to display */
#define min(a, b) ((a) < (b) ? (a) : (b))

static double alpha, beta;
static double A[N][N];
static double u1[N], v1[N], u2[N], v2[N], w[N], x[N], y[N], z[N];

int main(int argc, char** argv) {
	double start, stop;
	int i, j;

	acc_init(acc_device_default);
	
	start = omp_get_wtime();

	/* Initialize the array */
	alpha = 1.5;
	beta = 1.2;

	double fn = (double)N;

  #pragma acc data create(A, x, y, z, u1, v1, u2, v2), copyout(w)
  {
    #pragma acc kernels
    {
	for (i = 0; i < N; i++) {
		u1[i] = i;
		u2[i] = ((i + 1) / fn) / 2.0;
		v1[i] = ((i + 1) / fn) / 4.0;
		v2[i] = ((i + 1) / fn) / 6.0;
		y[i] = ((i + 1) / fn) / 8.0;
		z[i] = ((i + 1) / fn) / 9.0;
		x[i] = 0.0;
		w[i] = 0.0;
		#pragma acc loop vector
		for (j = 0; j < N; j++) 
			A[i][j] = (double) (i * j % N) / N;
	}

	/* Kernel (Vector Multiplication and Matrix Addition) */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			x[i] = x[i] + beta * A[j][i] + y[j];
		}
	}

	for (i = 0; i < N; i++) {
		x[i] = x[i] + z[i];
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			w[i]= w[i] + alpha * A[i][j] * x[j];
		}
	}

    }
  }

	stop = omp_get_wtime();

	/* Print the result vector, or at least part of it */
	for (i = 0; i < min(N, PRINT); i++) {
		fprintf(stderr, "%0.2lf ", w[i]);
	}
	fprintf(stderr, "\n");
	
	printf("Elapsed time: %2.2lf seconods\n", stop - start);

	return 0;
}
