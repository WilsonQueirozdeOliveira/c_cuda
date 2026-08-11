#include <stdio.h>
#include <cuda_runtime.h>

#define N 512

// Macro para checar erros CUDA de forma limpa
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

__global__ void add_vectors(const int *a, const int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // Inicializa vetores no host
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Aloca memória na GPU
    CHECK_CUDA(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    // Copia dados do host para a GPU
    CHECK_CUDA(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    // Lança o kernel (1 bloco com N threads)
    add_vectors<<<1, N>>>(dev_a, dev_b, dev_c, N);

    // Verifica erros de lançamento do kernel
    CHECK_CUDA(cudaGetLastError());

    // Espera a GPU terminar e verifica erros de execução
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copia resultado de volta para o host
    CHECK_CUDA(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verifica alguns resultados
    printf("Primeiros 10 resultados:\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %d + %d = %d\n", i, a[i], b[i], c[i]);
    }

    printf("\nUltimos 5 resultados:\n");
    for (int i = N - 5; i < N; i++) {
        printf("c[%d] = %d + %d = %d\n", i, a[i], b[i], c[i]);
    }

    // Libera memória da GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    printf("\nSucesso! Soma de vetores concluida.\n");
    return 0;
}
