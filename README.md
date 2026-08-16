# c_cuda

Repositório de exemplos progressivos em **CUDA C/C++**.

## Exemplos

| Arquivo     | Descrição                                                              |
|-------------|------------------------------------------------------------------------|
| `cuda_1.cu` | Kernel vazio + `printf` (teste mínimo)                                 |
| `cuda_2.cu` | Soma de dois números na GPU + error checking                           |
| `cuda_3.cu` | Soma de vetores + tratamento completo de erros                         |
| `cuda_4.cu` | Multiplicação de matrizes (Global Memory) + timing                     |
| `cuda_5.cu` | Multiplicação de matrizes com **Shared Memory** (Tiled)                |
| `cuda_6.cu` | **Comparação de performance**: Global Memory vs Shared Memory          |
| `cuda_7.cu` | **Parallel Reduction** com Shared Memory                               |
| `cuda_8.cu` | **CUDA Streams** e overlap CPU/GPU                                     |

## Compilação (automática)

```bash
make
```

O makefile detecta automaticamente a arquitetura da GPU (Windows + Linux/WSL).

```bash
make ARCH=sm_50    # forçar MX130
make ARCH=sm_75    # forçar GTX 1660
```

## Execução

**Linux / WSL:** `./cuda_6` `./cuda_7` `./cuda_8`  
**Windows:** `.\cuda_6.exe` `.\cuda_7.exe` `.\cuda_8.exe`

## Resultados reais – Comparação de GPUs

| Exemplo | Métrica | MX130 (sm_50) WSL | GTX 1660 (sm_75) WSL |
|---------|---------|-------------------|----------------------|
| **cuda_6** | Global Memory | 51.1 ms | 8.4 ms |
| **cuda_6** | Shared Memory | 20.4 ms | 4.1 ms |
| **cuda_6** | Speedup Shared | 2.51× | 2.06× |
| **cuda_7** | Reduction (1M elems) | ~5–6 ms | 1.4 ms |
| **cuda_8** | Sem overlap | 21.3 ms | 6.8 ms |
| **cuda_8** | Com streams | 17.8 ms | 4.4 ms |
| **cuda_8** | Speedup streams | 1.20× | 1.53× |

A GTX 1660 é ~4–6× mais rápida que a MX130 nos kernels testados.

## WSL vs Windows nativo (mesma GTX 1660)

| Exemplo | Métrica | WSL | Windows nativo |
|---------|---------|-----|----------------|
| **cuda_6** | Global Memory | 8.43 ms | **7.90 ms** |
| **cuda_6** | Shared Memory | 4.10 ms | 4.10 ms |
| **cuda_6** | Speedup Shared | 2.06× | 1.93× |
| **cuda_7** | Reduction | 1.41 ms | **0.20 ms** |
| **cuda_8** | Sem overlap | 6.78 ms | **4.97 ms** |
| **cuda_8** | Com streams | 4.44 ms | 4.52 ms |
| **cuda_8** | Speedup streams | **1.53×** | 1.10× |

**Conclusão:** kernels de compute ficam parecidos; Windows nativo tende a ser um pouco mais rápido em transferências e kernels muito curtos. WSL continua excelente para desenvolvimento.

## Máquinas testadas

| GPU | Compute Capability | Arquitetura | Ambiente |
|-----|--------------------|-------------|----------|
| **GTX 1660** | 7.5 | Turing | Desktop (WSL2 + Windows nativo) |
| **GeForce MX130** | 5.0 | Maxwell | Notebook (WSL2 Ubuntu 22.04) |

## Conceitos cobertos

| Exemplo | Conceito principal |
|---------|--------------------|
| cuda_1–2 | Kernel básico + error checking |
| cuda_3 | Soma de vetores |
| cuda_4 | Matrix multiplication (global memory) |
| cuda_5–6 | Shared Memory + comparação de performance |
| cuda_7 | Parallel Reduction com Shared Memory |
| cuda_8 | CUDA Streams + overlap de transferência |

## Requisitos

- NVIDIA GPU
- CUDA Toolkit (`nvcc`)
- **Windows**: Visual Studio Build Tools + `make` (winget install ezwinports.make)
- **Linux/WSL**: Ubuntu 22.04 recomendado

## Progresso

### Concluído
- [x] Modelo de execução CUDA (Host vs Device, Grid/Block/Thread)
- [x] Primeiro kernel + `<<<>>>`
- [x] `cudaMalloc` / `cudaMemcpy` / `cudaFree`
- [x] Tratamento de erros CUDA
- [x] Indexação de threads 1D e 2D
- [x] Soma de vetores
- [x] Multiplicação de matrizes (Global Memory)
- [x] Medição de tempo com `cudaEvent`
- [x] Hierarquia de memória (Registers, Shared, Global, Constant)
- [x] Shared Memory + Tiled Matrix Multiplication
- [x] Comparação de performance Global vs Shared
- [x] Parallel Reduction com Shared Memory
- [x] CUDA Streams e overlap CPU/GPU
- [x] Makefile inteligente (auto-detecta GPU no Windows e Linux)
- [x] Benchmark em duas GPUs (MX130 e GTX 1660)
- [x] Comparação WSL vs Windows nativo

### Próximos objetivos
- [ ] Bank Conflicts (demonstração + correção)
- [ ] Occupancy e limites do hardware
- [ ] Reduction warp-level (`__shfl`)
- [ ] Prefix Sum / Scan
- [ ] Histogramas
- [ ] Stencil / convolução simples
- [ ] Unified Memory
- [ ] Cooperative Groups
- [ ] Integração com cuBLAS
- [ ] Profiling com Nsight Systems / Nsight Compute
- [ ] Multi-GPU básico
- [ ] CUDA Graphs (introdução)
