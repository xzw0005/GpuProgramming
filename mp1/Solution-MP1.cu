/*
 * Redheffer Matrix Computation
 * MP1 Sample Solution, Spring 2016, GPU Programming @ Auburn University
 *
 * Compile this with:
 * nvcc -O3 -Xcompiler=-fopenmp -o Solution-MP1 Solution-MP1.cu
 *
 * See https://en.wikipedia.org/wiki/Redheffer_matrix
 */

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 20005

int h_result[SIZE*SIZE];

__global__ void compute_entry(int *d_result);
void compute_result();
void verify_result();
void check(int row, int col, int expected);

int main(int argc, char **argv)
{
    double start_time, time_to_compute, time_to_verify;

    printf("Initializing CUDA runtime...\n");
    cudaDeviceSynchronize(); /* To avoid measuring startup time */

    printf("Computing %d x %d Redheffer matrix...\n", SIZE, SIZE);

    start_time = omp_get_wtime();
    compute_result();
    time_to_compute = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    verify_result();
    time_to_verify = omp_get_wtime() - start_time;

    printf("Done (%2.3f seconds to compute, %2.3f seconds to verify)\n",
        time_to_compute, time_to_verify);
    return EXIT_SUCCESS;
}

__global__ void compute_entry(int *d_result)
{
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if (row >= SIZE || col >= SIZE)
        return;

    int i = row + 1;
    int j = col + 1;
    if (j == 1 || j % i == 0)
        d_result[row*SIZE + col] = 1;
    else
        d_result[row*SIZE + col] = 0;
}

/* Fills the h_result array with the SIZE x SIZE Redheffer matrix */
void compute_result()
{
    #define N 16
    dim3 grid_dim((SIZE+N-1)/N, (SIZE+N-1)/N, 1);
    dim3 block_dim(N, N, 1);
    int *d_result;
    assert(cudaMalloc(&d_result, sizeof(h_result)) == cudaSuccess);
    compute_entry<<<grid_dim, block_dim>>>(d_result);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemcpy(h_result, d_result, sizeof(h_result), cudaMemcpyDeviceToHost);
}

/* Verifies that the data in the h_result array is correct */
void verify_result()
{
    int row, col, i, j, expected;
    for (row = 0; row < SIZE; row++)
    {
        for (col = 0; col < SIZE; col++)
        {
            i = row + 1;
            j = col + 1;
            expected = (j == 1 || j % i == 0);
            check(row, col, expected);
        }
    }
}

/* Exits with an error message iff h_result[row*SIZE + col] != expected */
void check(int row, int col, int expected)
{
    if (h_result[row*SIZE + col] != expected)
    {
        fprintf(stderr, "Row %d column %d is incorrect.\n", row, col);
        fprintf(stderr, "  Should be:   %d\n", expected);
        fprintf(stderr, "  Is actually: %d\n", h_result[row*SIZE + col]);
        exit(EXIT_FAILURE);
    }
}
