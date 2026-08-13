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

**Linux / WSL:** `./cuda_8`  
**Windows:** `.\cuda_8.exe`

### Resultados reais (MX130)

**cuda_6 – Shared Memory vs Global**
```
Global Memory           | 51.125 ms
Shared Memory (Tiled)   | 20.399 ms
Speedup                 | 2.51x
```

**cuda_7 – Parallel Reduction**
```
Elementos: 1.048.576
Tempo do kernel: ~5–6 ms
Resultado: correto
```

**cuda_8 – Streams e Overlap**
```
Tempo SEM overlap: 21.323 ms
Tempo COM streams: 17.840 ms
Speedup: 1.20x
```

## Máquinas testadas

| GPU              | Compute Capability | Arquitetura | Ambiente                    |
|------------------|--------------------|-------------|-----------------------------|
| **GTX 1660**     | 7.5                | Turing      | Desktop (Windows)           |
| **GeForce MX130**| 5.0                | Maxwell     | Notebook (WSL2 Ubuntu 22.04)|

## Conceitos cobertos

| Exemplo   | Conceito principal                          |
|-----------|---------------------------------------------|
| cuda_1–2  | Kernel básico + error checking              |
| cuda_3    | Soma de vetores                             |
| cuda_4    | Matrix multiplication (global memory)       |
| cuda_5–6  | Shared Memory + comparação de performance   |
| cuda_7    | Parallel Reduction com Shared Memory        |
| cuda_8    | CUDA Streams + overlap de transferência     |

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
