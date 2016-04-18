/*
 * Redheffer Matrix Computation
 * MP1, Spring 2016, GPU Programming @ Auburn University
 * @author: Xing Wang
 *
 * Compile this with:
 * nvcc -O3 -Xcompiler=-fopenmp -o redheffer redheffer.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define CUDA_CHECK(e) { \
	cudaError_t err = (e); \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error: %s, line %d, %s: %s\n", __FILE__, __LINE__, #e, cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); \
	} \
}


#define SIZE 20000

int h_result[SIZE * SIZE];
void verify_result();
void check(int row, int col, int expected); 

__global__ static void compute_result(int *h_result);

int main(int argc, char **argv) {
	double start_time, time_to_compute, time_to_verify;

	printf("Initializing CUDA runtime...\n");	
	cudaDeviceSynchronize();

	printf("Computing %d x %d Redheffer matrix...\n", SIZE, SIZE);


	size_t matsize = sizeof(int) * SIZE * SIZE;
	/*int *h_result = (int*)malloc(matsize);
	if (h_result == NULL) {
		fprintf(stderr, "Unable to allocate host memory\n");
		exit(EXIT_FAILURE);
	}*/

	int *d_result;
	start_time = omp_get_wtime();
	CUDA_CHECK(cudaMalloc((void**)&d_result, matsize));
	CUDA_CHECK(cudaMemcpy(d_result, h_result, matsize, cudaMemcpyHostToDevice));

	/*int threadsPerBlock = 256;
	int blocksPerGrid = (matsize + threadsPerBlock - 1) / threadsPerBlock;*/ 
	
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x, (SIZE + threadsPerBlock.y - 1)/threadsPerBlock.y);

	compute_result<<<blocksPerGrid, threadsPerBlock>>>(d_result);
	
	CUDA_CHECK(cudaDeviceSynchronize());
	
	CUDA_CHECK(cudaMemcpy(h_result, d_result, matsize, cudaMemcpyDeviceToHost));

	time_to_compute = omp_get_wtime() - start_time;

	start_time = omp_get_wtime();
	verify_result();
	time_to_verify = omp_get_wtime() - start_time;
	
	CUDA_CHECK(cudaFree(d_result));
	CUDA_CHECK(cudaDeviceReset());
	
	printf("Done (%2.3f seconds to compute, %2.3f seconds to verify)\n", time_to_compute, time_to_verify);
	return EXIT_SUCCESS;
}

/* Fills the result array with the SIZE X SIZE Redheffer matrix */
__global__ static void compute_result(int *A) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y; 
	if (row < SIZE && col < SIZE) {
		int i = row + 1;
		int j = col + 1;
		if (j == 1 || j % i == 0)
			A[row * SIZE + col] = 1;
		else
			A[row * SIZE + col] = 0;
	}
}


/* Verifies that the data in the h_result array is correct */
void verify_result() {
	int row, col, i, j, expected;
	for (row = 0; row < SIZE; row++) {
		for (col = 0; col < SIZE; col++) {
			i = row + 1;
			j = col + 1;
			expected = (j == 1 || j % i == 0);
			check(row, col, expected);
		}
	}
}

/* Exits with an error message iff h_result[row * SIZE + col] != expected */
void check(int row, int col, int expected) {
	if (h_result[row * SIZE + col] != expected) {
		fprintf(stderr, "Row %d column %d is incorrect.\n", row, col);
		fprintf(stderr, "  Should be:	%d\n", expected);
		fprintf(stderr, "  Is actually:	%d\n", h_result[row * SIZE + col]);
		exit(EXIT_FAILURE);
	}
}

