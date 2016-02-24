/*
 * CUDA Vector Addition Kernel
 * J. Overbey (Spring 2016), based on NVIDIA_CUDA-6.0_Samples/0_Simple/vectorAdd
 *
 * Compile this with:
 *     nvcc -o vecAdd-CUDA vecAdd-CUDA.cu
 *
 * This program generates two random vectors (A and B) with NUM_ELTS elements
 * each, computes their sum on the device (GPU) using CUDA, and then verifies
 * the result by adding the vectors again on the CPU and comparing the results.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Number of elements in the vectors to be added */
#define NUM_ELTS 10000

/* Number of CUDA threads per block */
#define THREADS_PER_BLOCK 256

/* Maximum difference allowed between the GPU and CPU result vectors */
#define EPSILON 1e-5

/* If a CUDA call fails, display an error message and exit */
#define CUDA_CHECK(e) { \
	cudaError_t err = (e); \
	if (err != cudaSuccess) \
	{ \
		fprintf(stderr, "CUDA error: %s, line %d, %s: %s\n", \
			__FILE__, __LINE__, #e, cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); \
	} \
}

/*
 * Vector addition kernel.  Takes as input two arrays A and B, each with
 * NUM_ELTS elements, and stores their sum in C.
 */
__global__ static void vecAdd(const float *A, const float *B, float *C);

int main(void)
{
	printf("Adding %d-element vectors\n", NUM_ELTS);

	/* Create vectors on host; fill A and B with random numbers */
	size_t size = NUM_ELTS * sizeof(float);
	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Unable to allocate host memory\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < NUM_ELTS; ++i)
	{
		h_A[i] = rand()/(float)RAND_MAX;
		h_B[i] = rand()/(float)RAND_MAX;
	}

	/* Allocate device global memory for vectors */
	float *d_A, *d_B, *d_C;
	CUDA_CHECK(cudaMalloc((void **)&d_A, size));
	CUDA_CHECK(cudaMalloc((void **)&d_B, size));
	CUDA_CHECK(cudaMalloc((void **)&d_C, size));

	/* Copy vectors to device global memory */
	CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	/* Launch the CUDA kernel */
	int blocksPerGrid = (NUM_ELTS+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
	printf("Launching CUDA Kernel (%d blocks, %d threads/block)\n",
		blocksPerGrid, THREADS_PER_BLOCK);
	vecAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, d_C);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	/* Copy result vector from device global memory back to host memory */
	CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

	/* Verify that the result vector is correct */
	for (int i = 0; i < NUM_ELTS; ++i)
	{
		if (fabsf(h_A[i] + h_B[i] - h_C[i]) > EPSILON)
		{
			fprintf(stderr, "Result element %d is incorrect\n", i);
			fprintf(stderr, "  h_A[i] + h_B[i] = %f\n", h_A[i] + h_B[i]);
			fprintf(stderr, "           h_C[i] = %f\n", h_C[i]);
			fprintf(stderr, "  Difference is %f\n", fabsf(h_A[i] + h_B[i] - h_C[i]));
			fprintf(stderr, "     EPSILON is %f\n", EPSILON);
			exit(EXIT_FAILURE);
		}
	}

	/* Free device global memory */
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));

	/* Free host memory */
	free(h_A);
	free(h_B);
	free(h_C);

	/* Reset the device (unnecessary if not profiling, but good practice) */
	CUDA_CHECK(cudaDeviceReset());

	printf("Done\n");
	return 0;
}

__global__ static void vecAdd(const float *A, const float *B, float *C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < NUM_ELTS)
	{
		C[i] = A[i] + B[i];
	}
}
