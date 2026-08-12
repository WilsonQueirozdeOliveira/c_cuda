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

## Compilação (automática)

O `makefile` detecta automaticamente a arquitetura da GPU:

```bash
make
```

Ele mostra a arquitetura detectada e compila todos os exemplos.

### Forçar uma arquitetura específica

```bash
make ARCH=sm_50    # MX130
make ARCH=sm_75    # GTX 1660 / RTX 20xx
```

## Execução

**Linux / WSL:**
```bash
./cuda_6
```

**Windows:**
```powershell
.\cuda_6.exe
```

### Saída esperada de `cuda_6` (exemplo MX130):

```
================================================
  Comparacao de Performance - Matriz 1024x1024
================================================

Versao                  | Tempo (ms)
------------------------|-----------
Global Memory           |   51.125
Shared Memory (Tiled)   |   20.399
------------------------|-----------

Speedup (Shared / Global): 2.51x mais rapido

Verificacao: C[0][0] = 1024.0 (esperado: 1024.0)
Resultado correto!
================================================
```

## Máquinas testadas

| GPU              | Compute Capability | Arquitetura | Observação                  |
|------------------|--------------------|-------------|-----------------------------|
| **GTX 1660**     | 7.5                | Turing      | Desktop (Windows)           |
| **GeForce MX130**| 5.0                | Maxwell     | Notebook (WSL2 Ubuntu 22.04)|

## Destaques dos exemplos avançados

### `cuda_5.cu` e `cuda_6.cu` — Shared Memory
- Tiled Matrix Multiplication
- Uso de `__shared__` + `__syncthreads()`
- Tile 16×16
- Comparação lado a lado com Global Memory

### Por que Shared Memory é importante?

A memória global tem latência alta. A Shared Memory é muito mais rápida (fica dentro do SM).  
Ao carregar tiles, reduzimos drasticamente os acessos lentos à memória global.

## Requisitos

- NVIDIA GPU
- CUDA Toolkit (`nvcc` no PATH)
- **Windows**: Visual Studio Build Tools + `cl.exe`
- **Linux/WSL**: apenas o CUDA Toolkit (recomendado Ubuntu 22.04)

## Estrutura do Projeto

```
c_cuda/
├── cuda_1.cu      # Kernel vazio
├── cuda_2.cu      # Soma simples + erros
├── cuda_3.cu      # Soma de vetores
├── cuda_4.cu      # Matmul (Global Memory)
├── cuda_5.cu      # Matmul (Shared Memory)
├── cuda_6.cu      # Comparação Global vs Shared
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
- [ ] Reduction com Shared Memory
- [ ] Streams e overlap CPU/GPU
