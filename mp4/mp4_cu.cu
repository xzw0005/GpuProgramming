// Brute-force Key Search - TEA Encryption with 31-bit Key
// MP4, Spring 2016, GPU Programming @ Auburn University
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* Data to test with (this should be easy to change) */
const uint32_t orig_data[2] = { 0xDEADBEEF, 0x0BADF00D };
const uint32_t encrypted[2] = { 0xFF305F9B, 0xB9BDCECE  };

void encrypt(uint32_t *data, const uint32_t *key) {
    uint32_t v0=data[0], v1=data[1], sum=0, i;             /* set up */
    uint32_t delta=0x9e3779b9;                             /* a key schedule constant */
    uint32_t k0=key[0], k1=key[1], k2=key[2], k3=key[3];   /* cache key */
    for (i=0; i < 32; i++) {                               /* basic cycle start */
        sum += delta;
        v0 += ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
        v1 += ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
    }                                                      /* end cycle */
    data[0]=v0; data[1]=v1;
}

__global__ static void encryptGpu(uint32_t *data, uint32_t *key, uint32_t *key_out) {
    /* Try every possible 28-bit integer... */
    const uint32_t encrypted[2] = { 0xFF305F9B, 0xB9BDCECE};
	uint32_t k = blockDim.x * blockDim.y + threadIdx.x;
	if (k <= 0x0FFFFFFF) {
		uint32_t v0 = data[0], v1 = data[1], sum = 0, i;             /* set up */
		uint32_t delta=0x9e3779b9;                             /* a key schedule constant */
		uint32_t k0 = k + key[0] * 2048 * 1024, k1 = k + key[1] * 2048 * 1024, k2 = k + key[2] * 2048 * 1024, k3 = k + key[3] * 2048 * 1024;
		for (i=0; i < 32; i++) {                               /* basic cycle start */
			sum += delta;
			v0 += ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
			v1 += ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
		}                                                      /* end cycle */
        /* Did we get the correct encrypted values? */
        if (v0 == encrypted[0] && v1 == encrypted[1]) {
            key_out[0] = k0; key_out[1] = k1; key_out[2] = k2; key_out[3] = k3;
        }
    }	
}

void decrypt(uint32_t *data, const uint32_t *key) {
    uint32_t v0=data[0], v1=data[1], sum=0xC6EF3720, i;  /* set up */
    uint32_t delta=0x9e3779b9;                           /* a key schedule constant */
    uint32_t k0=key[0], k1=key[1], k2=key[2], k3=key[3]; /* cache key */
    for (i=0; i<32; i++) {                               /* basic cycle start */
        v1 -= ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
        v0 -= ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
        sum -= delta;
    }                                                    /* end cycle */
    data[0]=v0; data[1]=v1;
}

int main() {
	size_t sizeKey = 4 * sizeof(uint32_t);
	size_t sizeData = 2 * sizeof(uint32_t);
    uint32_t *key = (uint32_t *)malloc(sizeKey);
	uint32_t *data = (uint32_t *)malloc(sizeData);

    printf("Starting (this may take a while)...\n");
    double start = omp_get_wtime();
/////////////////////////////////////////////////////////////////
	uint32_t *d_key, *d_data, *d_keyout;
	cudaMalloc((void **)&d_key, sizeKey);
	cudaMalloc((void **)&d_data, sizeData);
	cudaMalloc((void **)&d_keyout, sizeKey);
	for (uint32_t n = 0; n < 128; n++) {
	    key[0] = key[1] = key[2] = key[3] = n;
	    cudaMemcpy(d_key, key, sizeKey, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_data, orig_data, sizeData, cudaMemcpyHostToDevice);
	    encryptGpu<<<2048, 1024>>>(d_data, d_key, d_keyout);
	}
	
	
	cudaDeviceSynchronize();
	cudaMemcpy(key, d_keyout, sizeKey, cudaMemcpyDeviceToHost);
	
	cudaFree(d_data);
	cudaFree(d_key);
////////////////////////////////////////////////////////////////
    printf("Elapsed time: %f seconds\n", omp_get_wtime() - start);

    /* Assume the above loop will find a key */
    printf("Found key: (hexadecimal) %08x %08x %08x %08x\n", key[0], key[1], key[2], key[3]);
    data[0] = orig_data[0];
    data[1] = orig_data[1];
    printf("The original values are (hexadecimal):  %08x %08x\n", data[0], data[1]);
    encrypt(data, key);
    printf("The encrypted values are (hexadecimal): %08x %08x\n", data[0], data[1]);
    printf("They should be:                         %08x %08x\n", encrypted[0], encrypted[1]);
    if (data[0] == encrypted[0] && data[1] == encrypted[1]) {
        printf("SUCCESS!\n");
        return 0;
    } else {
        printf("FAILED\n");
        return 1;
    }
}