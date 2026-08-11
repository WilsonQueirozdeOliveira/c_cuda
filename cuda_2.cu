#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    int c;
    int *dev_c;

    cudaError_t err;

    err = cudaMalloc((void**)&dev_c, sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    add<<<1,1>>>(2, 7, dev_c);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("2 + 7 = %d\n", c);

    cudaFree(dev_c);
    return 0;
}
