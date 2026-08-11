#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024          // Tamanho da matriz (N x N)
#define TILE 16         // Tamanho do tile (16x16 threads por bloco)

// Kernel otimizado com Shared Memory (Tiled Matrix Multiplication)
__global__ void matmul_shared(float *A, float *B, float *C, int n) {
    // Memória compartilhada para os tiles
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Percorre os tiles
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        // Carrega tile de A para shared memory
        if (row < n && (t * TILE + threadIdx.x) < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Carrega tile de B para shared memory
        if (col < n && (t * TILE + threadIdx.y) < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // Sincroniza para garantir que todos os dados foram carregados
        __syncthreads();

        // Calcula o produto parcial usando a shared memory (muito mais rápido)
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Sincroniza antes de carregar o próximo tile
        __syncthreads();
    }

    // Escreve o resultado
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Inicializa matrizes com 1.0 (resultado esperado = N)
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Medição de tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_shared<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verificação
    printf("Multiplicacao de matrizes %dx%d com Shared Memory (TILE=%d)\n", N, N, TILE);
    printf("C[0][0]     = %.1f (esperado: %d.0)\n", h_C[0], N);
    printf("C[0][1]     = %.1f\n", h_C[1]);
    printf("C[%d][%d] = %.1f\n", N-1, N-1, h_C[(N-1)*N + (N-1)]);
    printf("Tempo do kernel: %.3f ms\n", ms);

    bool ok = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - (float)N) > 1e-3) {
            ok = false;
            break;
        }
    }

    if (ok) printf("Sucesso! Shared Memory funcionando.\n");
    else    printf("Erro na verificacao.\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
