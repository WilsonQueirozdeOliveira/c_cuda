# c_cuda

Repositório de exemplos progressivos em **CUDA C/C++**.

## Exemplos

| Arquivo     | Descrição                                                              |
|-------------|------------------------------------------------------------------------|
| `cuda_1.cu` | Kernel vazio + `printf` (teste mínimo)                                 |
| `cuda_2.cu` | Soma de dois números na GPU + error checking                           |
| `cuda_3.cu` | Soma de vetores + tratamento completo de erros                         |
| `cuda_4.cu` | Multiplicação de matrizes (versão global memory) + timing              |
| `cuda_5.cu` | **Multiplicação de matrizes com Shared Memory (Tiled)** — otimização   |

## Compilação

```powershell
# Compilar todos
make

# Ou individualmente
nvcc -o cuda_5.exe cuda_5.cu
```

## Execução

```powershell
.\cuda_5.exe
```

### Saída esperada de `cuda_5`:

```
Multiplicacao de matrizes 1024x1024 com Shared Memory (TILE=16)
C[0][0]     = 1024.0 (esperado: 1024.0)
C[0][1]     = 1024.0
C[1023][1023] = 1024.0
Tempo do kernel: X.XXX ms
Sucesso! Shared Memory funcionando.
```

## Destaques do `cuda_5.cu` (Shared Memory)

- **Tiled Matrix Multiplication** (técnica clássica de otimização)
- Uso de `__shared__` memory (muito mais rápida que a global)
- `__syncthreads()` para sincronização dentro do bloco
- Tile de 16×16 threads
- Matriz 1024×1024 (bem maior que o exemplo anterior)
- Medição de tempo com `cudaEvent`

### Por que Shared Memory é importante?

A memória global tem latência alta. A Shared Memory é ~100× mais rápida e fica dentro do SM (Streaming Multiprocessor).  
Ao carregar tiles (blocos) para a shared memory, reduzimos drasticamente o número de acessos à memória global.

## Requisitos

- NVIDIA GPU
- CUDA Toolkit (`nvcc` no PATH)
- Visual Studio Build Tools (Windows) — necessário para o `cl.exe`

## Estrutura do Projeto

```
c_cuda/
├── cuda_1.cu      # Kernel vazio
├── cuda_2.cu      # Soma simples + erros
├── cuda_3.cu      # Soma de vetores
├── cuda_4.cu      # Matmul (global memory)
├── cuda_5.cu      # Matmul otimizado (Shared Memory)
├── makefile
└── README.md
```

## Próximos passos (sugestões)

- [x] Tratamento de erros CUDA
- [x] Soma de vetores
- [x] Multiplicação de matrizes
- [x] Medição de tempo (`cudaEvent`)
- [x] Shared Memory (otimização)
- [ ] Comparação de performance (Global vs Shared)
- [ ] Reduction com Shared Memory
- [ ] Streams e overlap CPU/GPU
