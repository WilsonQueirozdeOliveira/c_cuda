# c_cuda

Repositório de exemplos básicos em **CUDA C/C++**.

## Exemplos

| Arquivo     | Descrição                                                      |
|-------------|----------------------------------------------------------------|
| `cuda_1.cu` | Kernel vazio + `printf` (teste mínimo)                         |
| `cuda_2.cu` | Soma de dois números na GPU (`2 + 7`) + error checking         |
| `cuda_3.cu` | Soma de vetores + tratamento completo de erros                 |
| `cuda_4.cu` | **Multiplicação de matrizes** + medição de tempo (`cudaEvent`) |

## Compilação

```powershell
# Compilar todos os exemplos
make

# Ou individualmente
nvcc -o cuda_1.exe cuda_1.cu
nvcc -o cuda_2.exe cuda_2.cu
nvcc -o cuda_3.exe cuda_3.cu
nvcc -o cuda_4.exe cuda_4.cu
```

## Execução

```powershell
.\cuda_1.exe
.\cuda_2.exe
.\cuda_3.exe
.\cuda_4.exe
```

### Saída esperada de `cuda_4`:

```
Multiplicacao de matrizes 32x32
C[0][0] = 64.0 (esperado: 64.0)
C[0][1] = 64.0
C[31][31] = 64.0

Tempo de execucao do kernel: 0.XXX ms

Sucesso!
```

## Destaques do `cuda_4.cu`

- Multiplicação de matrizes 32×32
- Grid 2D (`dim3`)
- Medição precisa de tempo com `cudaEventRecord` + `cudaEventElapsedTime`
- Macro `CHECK_CUDA` para tratamento de erros

## Requisitos

- NVIDIA GPU
- CUDA Toolkit instalado (`nvcc` disponível no PATH)
- Visual Studio Build Tools (Windows) — necessário para o `cl.exe`

### Instalação no Windows

1. Baixe o [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Instale o [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) com a workload **Desktop development with C++**
3. Atualize o driver NVIDIA se necessário (`nvidia-smi`)

## Estrutura do Projeto

```
c_cuda/
├── cuda_1.cu      # Exemplo 1 - Kernel vazio
├── cuda_2.cu      # Exemplo 2 - Soma simples + erros
├── cuda_3.cu      # Exemplo 3 - Soma de vetores + erros
├── cuda_4.cu      # Exemplo 4 - Multiplicação de matrizes + timing
├── makefile       # Compilação automática
└── README.md
```

## Próximos passos (sugestões)

- [x] Adicionar tratamento de erros CUDA (`cudaGetLastError`)
- [x] Exemplo de soma de vetores
- [x] Exemplo de multiplicação de matrizes
- [x] Medição de tempo com `cudaEvent`
- [ ] Shared memory (otimização)
- [ ] Matrizes maiores + comparação CPU vs GPU
