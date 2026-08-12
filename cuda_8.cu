#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)          // 1 milhão de elementos por stream
#define NSTREAMS 4           // Número de streams
#define BLOCK_SIZE 256

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    // Aloca memória pinned (page-locked) no host → necessário para overlap real
    float *h_a, *h_b, *h_c;
    CHECK_CUDA(cudaMallocHost(&h_a, NSTREAMS * bytes));
    CHECK_CUDA(cudaMallocHost(&h_b, NSTREAMS * bytes));
    CHECK_CUDA(cudaMallocHost(&h_c, NSTREAMS * bytes));

    // Inicializa dados
    for (int i = 0; i < NSTREAMS * N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Aloca memória na GPU (uma região grande)
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, NSTREAMS * bytes));
    CHECK_CUDA(cudaMalloc(&d_b, NSTREAMS * bytes));
    CHECK_CUDA(cudaMalloc(&d_c, NSTREAMS * bytes));

    // Cria streams
    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Eventos para medir tempo
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;

    // ========== Versão SEM overlap (sequencial) ==========
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < NSTREAMS; i++) {
        float *ha = h_a + i * N;
        float *hb = h_b + i * N;
        float *hc = h_c + i * N;
        float *da = d_a + i * N;
        float *db = d_b + i * N;
        float *dc = d_c + i * N;

        CHECK_CUDA(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
        vector_add<<<blocks, threads>>>(da, db, dc, N);
        CHECK_CUDA(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_seq = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_seq, start, stop));

    // ========== Versão COM streams (overlap) ==========
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < NSTREAMS; i++) {
        float *ha = h_a + i * N;
        float *hb = h_b + i * N;
        float *hc = h_c + i * N;
        float *da = d_a + i * N;
        float *db = d_b + i * N;
        float *dc = d_c + i * N;

        // Todas as operações vão para streams diferentes → podem se sobrepor
        CHECK_CUDA(cudaMemcpyAsync(da, ha, bytes, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(db, hb, bytes, cudaMemcpyHostToDevice, streams[i]));
        vector_add<<<blocks, threads, 0, streams[i]>>>(da, db, dc, N);
        CHECK_CUDA(cudaMemcpyAsync(hc, dc, bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Espera todos os streams terminarem
    for (int i = 0; i < NSTREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_overlap = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_overlap, start, stop));

    // Verificação
    bool ok = true;
    for (int i = 0; i < NSTREAMS * N; i++) {
        if (fabs(h_c[i] - 3.0f) > 1e-5) {
            ok = false;
            break;
        }
    }

    printf("========================================\n");
    printf("  Streams e Overlap CPU/GPU\n");
    printf("========================================\n");
    printf("Elementos por stream: %d\n", N);
    printf("Número de streams: %d\n", NSTREAMS);
    printf("Total de elementos: %d\n", NSTREAMS * N);
    printf("\n");
    printf("Tempo SEM overlap (sequencial): %.3f ms\n", time_seq);
    printf("Tempo COM streams (overlap):    %.3f ms\n", time_overlap);
    printf("Speedup: %.2fx\n", time_seq / time_overlap);
    printf("\n");
    printf("Verificacao: c[0] = %.1f (esperado: 3.0)\n", h_c[0]);
    if (ok) printf("Resultado correto!\n");
    else    printf("Erro no resultado.\n");
    printf("========================================\n");

    // Limpeza
    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);

    return 0;
}
