/*
 * OpenACC Vector Addition Kernel
 * J. Overbey (Spring 2016), based on NVIDIA_CUDA-6.0_Samples/0_Simple/vectorAdd
 *
 * Compile this with:
 *     pgcc -ta=nvidia -Minfo=accel -o vecAdd-OpenACC vecAdd-OpenACC.c
 *
 * This program generates two random vectors (A and B) with NUM_ELTS elements
 * each, computes their sum on the device (GPU) using OpenACC, and then verifies
 * the result by adding the vectors again on the CPU and comparing the results.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Number of elements in the vectors to be added */
#define NUM_ELTS 10000

/* Maximum difference allowed between the GPU and CPU result vectors */
#define EPSILON 1e-5

int main(void)
{
	printf("Adding %d-element vectors\n", NUM_ELTS);

	/* Create vectors on host; fill A and B with random numbers */
	size_t size = NUM_ELTS * sizeof(float);
	float *A = (float *)malloc(size);
	float *B = (float *)malloc(size);
	float *C = (float *)malloc(size);
	if (A == NULL || B == NULL || C == NULL)
	{
		fprintf(stderr, "Unable to allocate host memory\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < NUM_ELTS; ++i)
	{
		A[i] = rand()/(float)RAND_MAX;
		B[i] = rand()/(float)RAND_MAX;
	}

	/* Copy A and B from CPU to GPU memory at the beginning of this block,
	   then copy C back from GPU to CPU memory at the end of this block */
	{
		/* Run the iterations of this loop concurrently on the GPU */
		for (int i = 0; i < NUM_ELTS; ++i)
		{
			C[i] = A[i] + B[i];
		}
	}

	/* Verify that the result vector is correct */
	for (int i = 0; i < NUM_ELTS; ++i)
	{
		if (fabsf(A[i] + B[i] - C[i]) > EPSILON)
		{
			fprintf(stderr, "Result element %d is incorrect\n", i);
			fprintf(stderr, "  A[i] + B[i] = %f\n", A[i] + B[i]);
			fprintf(stderr, "         C[i] = %f\n", C[i]);
			fprintf(stderr, "  Difference is %f\n", fabsf(A[i] + B[i] - C[i]));
			fprintf(stderr, "     EPSILON is %f\n", EPSILON);
			exit(EXIT_FAILURE);
		}
	}

	/* Free host memory */
	free(A);
	free(B);
	free(C);

	printf("Done\n");
	return 0;
}
