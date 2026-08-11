#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define TILE 16

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

// ==================== Versão 1: Global Memory ====================
__global__ void matmul_global(const float *A, const float *B, float *C, int n) {
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

// ==================== Versão 2: Shared Memory (Tiled) ====================
__global__ void matmul_shared(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        if (row < n && (t * TILE + threadIdx.x) < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && (t * TILE + threadIdx.y) < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float time_global = 0.0f, time_shared = 0.0f;

    // ========== 1. Global Memory ==========
    CHECK_CUDA(cudaEventRecord(start));
    matmul_global<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_global, start, stop));

    // ========== 2. Shared Memory ==========
    CHECK_CUDA(cudaEventRecord(start));
    matmul_shared<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_shared, start, stop));

    // Copia resultado (só para verificar)
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // ========== Resultados ==========
    printf("================================================\n");
    printf("  Comparacao de Performance - Matriz %dx%d\n", N, N);
    printf("================================================\n\n");

    printf("Versao                  | Tempo (ms)\n");
    printf("------------------------|-----------\n");
    printf("Global Memory           | %8.3f\n", time_global);
    printf("Shared Memory (Tiled)   | %8.3f\n", time_shared);
    printf("------------------------|-----------\n");

    float speedup = time_global / time_shared;
    printf("\nSpeedup (Shared / Global): %.2fx mais rapido\n", speedup);

    printf("\nVerificacao: C[0][0] = %.1f (esperado: %d.0)\n", h_C[0], N);

    if (fabs(h_C[0] - (float)N) < 1e-3)
        printf("Resultado correto!\n");
    else
        printf("Erro no resultado.\n");

    printf("================================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
