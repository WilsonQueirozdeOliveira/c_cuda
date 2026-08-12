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
| `cuda_7.cu` | **Parallel Reduction** com Shared Memory (soma de 1 milhão de elementos)|

## Compilação (automática)

O `makefile` detecta automaticamente a arquitetura da GPU:

```bash
make
```

### Forçar uma arquitetura específica

```bash
make ARCH=sm_50    # MX130
make ARCH=sm_75    # GTX 1660 / RTX 20xx
```

## Execução

**Linux / WSL:**
```bash
./cuda_7
```

**Windows:**
```powershell
.\cuda_7.exe
```

### Saída esperada de `cuda_7` (MX130):

```
========================================
  Reduction com Shared Memory
========================================
Elementos: 1048576
Blocos: 4096 | Threads/bloco: 256
Soma GPU: 1048576.0
Esperado: 1048576.0
Tempo do kernel: 5.001 ms
Resultado correto!
========================================
```

### Saída esperada de `cuda_6` (MX130):

```
Global Memory           | 51.125 ms
Shared Memory (Tiled)   | 20.399 ms
Speedup                 | 2.51x mais rápido
```

## Máquinas testadas

| GPU              | Compute Capability | Arquitetura | Observação                  |
|------------------|--------------------|-------------|-----------------------------|
| **GTX 1660**     | 7.5                | Turing      | Desktop (Windows)           |
| **GeForce MX130**| 5.0                | Maxwell     | Notebook (WSL2 Ubuntu 22.04)|

## Destaques dos exemplos avançados

### Shared Memory (cuda_5 e cuda_6)
- Tiled Matrix Multiplication
- `__shared__` + `__syncthreads()`
- Comparação Global vs Shared

### Parallel Reduction (cuda_7)
- Soma de 1 milhão de elementos
- Reduction em árvore dentro do bloco usando Shared Memory
- Técnica fundamental usada em muitas bibliotecas CUDA

## Requisitos

- NVIDIA GPU
- CUDA Toolkit (`nvcc` no PATH)
- **Windows**: Visual Studio Build Tools
- **Linux/WSL**: Ubuntu 22.04 recomendado

## Estrutura do Projeto

```
c_cuda/
├── cuda_1.cu ... cuda_7.cu
├── makefile       # Detecta arquitetura automaticamente
└── README.md
```

## Progresso

- [x] Tratamento de erros CUDA
- [x] Soma de vetores
- [x] Multiplicação de matrizes
- [x] Medição de tempo (`cudaEvent`)
- [x] Shared Memory (otimização)
- [x] Comparação de performance (Global vs Shared)
- [x] Makefile inteligente (auto-detecta GPU)
- [x] Parallel Reduction com Shared Memory
- [ ] Streams e overlap CPU/GPU
