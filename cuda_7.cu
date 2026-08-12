#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N (1 << 20)   // 1 milhão de elementos
#define BLOCK_SIZE 256

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

// Kernel de Reduction com Shared Memory
__global__ void reduce_shared(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cada thread carrega um elemento (ou zero se estiver fora)
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction em árvore dentro do bloco
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 de cada bloco escreve o resultado parcial
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_partial = (float*)malloc(((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(float));

    // Inicializa vetor com 1.0 → soma esperada = N
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_partial;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_partial, numBlocks * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Medição de tempo
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    reduce_shared<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Soma final na CPU dos resultados parciais dos blocos
    float sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partial[i];
    }

    printf("========================================\n");
    printf("  Reduction com Shared Memory\n");
    printf("========================================\n");
    printf("Elementos: %d\n", N);
    printf("Blocos: %d | Threads/bloco: %d\n", numBlocks, BLOCK_SIZE);
    printf("Soma GPU: %.1f\n", sum);
    printf("Esperado: %.1f\n", (float)N);
    printf("Tempo do kernel: %.3f ms\n", ms);

    if (fabs(sum - (float)N) < 1e-1)
        printf("Resultado correto!\n");
    else
        printf("Erro no resultado.\n");

    printf("========================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_partial);
    free(h_input);
    free(h_partial);

    return 0;
}
