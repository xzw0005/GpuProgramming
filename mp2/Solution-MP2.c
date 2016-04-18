/* **************************** SAMPLE SOLUTION **************************** */
/* ****************************  (Spring 2016)  **************************** */
/**
 * This version is stamped on Apr. 14, 2015
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */
/* Modified by Jeff Overbey for Auburn COMP 4960-017/7930-001 (Sp16) */
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>

#define N 8000    /* Matrix width/height */
#define PRINT 10  /* Number of rows/columns to display */
#define min(a,b) ((a) < (b) ? (a) : (b))

static double alpha, beta;
static double A[N][N];
static double u1[N], v1[N], u2[N], v2[N], w[N], x[N], y[N], z[N];

int main(int argc, char** argv)
{
  double start, stop;
  int i, j;

  acc_init(acc_device_default);

  start = omp_get_wtime();

  /* Initialize the array */
  alpha = 1.5;
  beta = 1.2;

  double fn = (double)N;

  #pragma acc data create(A,u1,u2,v1,v2,x,y,z) copyout(w[0:min(N,PRINT)])
  {

  #pragma acc parallel loop gang
  for (i = 0; i < N; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      #pragma acc loop vector
      for (j = 0; j < N; j++)
        A[i][j] = (double) (i*j % N) / N;
    }

  /* Kernel (Vector Multiplication and Matrix Addition) */
  #pragma acc parallel loop
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
    }
  }

  #pragma acc parallel loop vector_length(32)
  for (i = 0; i < N; i++) {
    double sum = x[i];
    for (j = 0; j < N; j++)
      sum += beta * A[j][i] * y[j];
    sum += z[i];
    x[i] = sum;
  }

  #pragma acc parallel loop
  for (i = 0; i < N; i++) {
    double sum = w[i];
    for (j = 0; j < N; j++)
      sum += alpha * A[i][j] * x[j];
    w[i] = sum;
  }

  }

  stop = omp_get_wtime();

  /* Print the result vector, or at least part of it */
  for (i = 0; i < min(N,PRINT); i++) {
    fprintf(stderr, "%0.2lf ", w[i]);
  }
  fprintf(stderr, "\n");

  printf("Elapsed time: %2.2lf seconds\n", stop - start);

  return 0;
}
