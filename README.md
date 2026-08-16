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

O makefile detecta automaticamente a arquitetura da GPU.

```bash
make ARCH=sm_50    # forçar MX130
make ARCH=sm_75    # forçar GTX 1660
```

## Execução

**Linux / WSL:** `./cuda_6` `./cuda_7` `./cuda_8`  
**Windows:** `.\cuda_6.exe` `.\cuda_7.exe` `.\cuda_8.exe`

## Resultados reais – Comparação de GPUs

| Exemplo | Métrica | MX130 (sm_50) | GTX 1660 (sm_75) |
|---------|---------|---------------|------------------|
| **cuda_6** | Global Memory | 51.1 ms | **8.4 ms** |
| **cuda_6** | Shared Memory | 20.4 ms | **4.1 ms** |
| **cuda_6** | Speedup Shared | 2.51× | 2.06× |
| **cuda_7** | Reduction (1M elems) | ~5–6 ms | **1.4 ms** |
| **cuda_8** | Sem overlap | 21.3 ms | **6.8 ms** |
| **cuda_8** | Com streams | 17.8 ms | **4.4 ms** |
| **cuda_8** | Speedup streams | 1.20× | **1.53×** |

A GTX 1660 é ~4–6× mais rápida que a MX130 nos kernels testados.

## Máquinas testadas

| GPU | Compute Capability | Arquitetura | Ambiente |
|-----|--------------------|-------------|----------|
| **GTX 1660** | 7.5 | Turing | Desktop (WSL2 / Windows) |
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
- **Windows**: Visual Studio Build Tools
- **Linux/WSL**: Ubuntu 22.04 recomendado

## Progresso

- [x] Tratamento de erros CUDA
- [x] Soma de vetores
- [x] Multiplicação de matrizes
- [x] Medição de tempo (`cudaEvent`)
- [x] Shared Memory
- [x] Comparação de performance
- [x] Makefile inteligente (auto-detecta GPU)
- [x] Parallel Reduction
- [x] Streams e overlap CPU/GPU
