#include <stdio.h>
#include <cuda_runtime.h>

#define N 32   // Matriz N x N (mantenha pequeno para caber em 1 bloco)

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

// Kernel de multiplicação de matrizes (C = A * B)
__global__ void matrix_mul(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(void) {
    const int size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Inicializa matrizes no host
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;          // Matriz A preenchida com 1
        h_B[i] = 2.0f;          // Matriz B preenchida com 2
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Configuração de grid e blocos (2D)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    // === Medição de tempo com cudaEvent ===
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    matrix_mul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copia resultado de volta
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Mostra alguns valores
    printf("Multiplicacao de matrizes %dx%d\n", N, N);
    printf("C[0][0] = %.1f (esperado: %.1f)\n", h_C[0], (float)N * 1.0f * 2.0f);
    printf("C[0][1] = %.1f\n", h_C[1]);
    printf("C[%d][%d] = %.1f\n", N-1, N-1, h_C[(N-1)*N + (N-1)]);

    printf("\nTempo de execucao do kernel: %.3f ms\n", milliseconds);

    // Limpeza
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nSucesso!\n");
    return 0;
}
