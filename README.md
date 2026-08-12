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

## Compilação

```powershell
# Compilar todos
make

# Ou individualmente
nvcc -o cuda_6.exe cuda_6.cu
```

## Execução

```powershell
.\cuda_1.exe
.\cuda_2.exe
.\cuda_3.exe
.\cuda_4.exe
.\cuda_5.exe
.\cuda_6.exe
```

### Saída esperada de `cuda_6`:

```
================================================
  Comparacao de Performance - Matriz 1024x1024
================================================

Versao                  | Tempo (ms)
------------------------|-----------
Global Memory           |   XX.XXX
Shared Memory (Tiled)   |    X.XXX
------------------------|-----------

Speedup (Shared / Global): X.XXx mais rapido

Verificacao: C[0][0] = 1024.0 (esperado: 1024.0)
Resultado correto!
================================================
```

## Destaques dos exemplos avançados

### `cuda_5.cu` — Shared Memory
- Tiled Matrix Multiplication (técnica clássica)
- Uso de `__shared__` + `__syncthreads()`
- Tile 16×16
- Matriz 1024×1024

### `cuda_6.cu` — Comparação de Performance
- Executa as duas versões (Global e Shared) na mesma matriz
- Mede o tempo de cada uma com `cudaEvent`
- Calcula o speedup automaticamente

### Por que Shared Memory é importante?

A memória global tem latência alta (~400-800 ciclos).  
A Shared Memory é ~100× mais rápida e fica dentro do SM.  
Ao carregar tiles para a Shared Memory, reduzimos drasticamente os acessos lentos à memória global.

## Requisitos

- NVIDIA GPU (testado em GTX 1660)
- CUDA Toolkit (`nvcc` no PATH)
- Visual Studio Build Tools (Windows) — necessário para o `cl.exe`

### Dica Windows (PATH permanente)

Adicione estas pastas nas Variáveis de Ambiente do sistema:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin
C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64
```

## Estrutura do Projeto

```
c_cuda/
├── cuda_1.cu      # Kernel vazio
├── cuda_2.cu      # Soma simples + erros
├── cuda_3.cu      # Soma de vetores
├── cuda_4.cu      # Matmul (Global Memory)
├── cuda_5.cu      # Matmul (Shared Memory)
├── cuda_6.cu      # Comparação Global vs Shared
├── makefile
└── README.md
```

## Progresso

- [x] Tratamento de erros CUDA
- [x] Soma de vetores
- [x] Multiplicação de matrizes
- [x] Medição de tempo (`cudaEvent`)
- [x] Shared Memory (otimização)
- [x] Comparação de performance (Global vs Shared)
- [ ] Reduction com Shared Memory
- [ ] Streams e overlap CPU/GPU
