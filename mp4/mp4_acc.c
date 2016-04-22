// Brute-force Key Search - TEA Encryption with 31-bit Key
// MP4, Spring 2016, GPU Programming @ Auburn University
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <openacc.h>

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
    uint32_t key[4]  = { 0, 0, 0, 0};
    uint32_t data[2] = { 0, 0 };

    printf("Starting (this may take a while)...\n");
    double start = omp_get_wtime();
    #pragma acc parallel loop
    for (uint32_t k = 0; k <= 0x0FFFFFFF; k++) {
       
        uint32_t v0 = orig_data[0], v1 = orig_data[1], sum=0, i; 
	uint32_t delta=0x9e3779b9;                             /* a key schedule constant */
        uint32_t k0 = k, k1 = k, k2 = k, k3 = k;
        
		for (i=0; i < 32; i++) {                               /* basic cycle start */
			sum += delta;
			v0 += ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
			v1 += ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
		}                                                      /* end cycle */

        if (v0 == encrypted[0] && v1 == encrypted[1]) {
			key[0] = k0; key[1] = k1; key[2] = k2; key[3] = k3;
        }
    }
    
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
