# c_cuda

Repositório de exemplos básicos em **CUDA C/C++**.

## Exemplos

| Arquivo     | Descrição                                              |
|-------------|--------------------------------------------------------|
| `cuda_1.cu` | Kernel vazio + `printf` (teste mínimo)                 |
| `cuda_2.cu` | Soma de dois números na GPU (`2 + 7`) + error checking |
| `cuda_3.cu` | **Soma de vetores** + tratamento completo de erros     |

## Compilação

```powershell
# Compilar todos os exemplos
make

# Ou individualmente
nvcc -o cuda_1.exe cuda_1.cu
nvcc -o cuda_2.exe cuda_2.cu
nvcc -o cuda_3.exe cuda_3.cu
```

## Execução

```powershell
.\cuda_1.exe
.\cuda_2.exe
.\cuda_3.exe
```

### Saída esperada de `cuda_3`:

```
Primeiros 10 resultados:
c[0] = 0 + 0 = 0
c[1] = 1 + 2 = 3
c[2] = 2 + 4 = 6
...

Ultimos 5 resultados:
c[507] = 507 + 1014 = 1521
...

Sucesso! Soma de vetores concluida.
```

## Destaques do `cuda_3.cu`

- Macro `CHECK_CUDA` para facilitar o tratamento de erros
- Uso de `cudaGetLastError()` após o lançamento do kernel
- `cudaDeviceSynchronize()` para capturar erros de execução
- Soma de vetores com 512 elementos

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
├── makefile       # Compilação automática
└── README.md
```

## Próximos passos (sugestões)

- [x] Adicionar tratamento de erros CUDA (`cudaGetLastError`)
- [x] Exemplo de soma de vetores
- [ ] Exemplo de multiplicação de matrizes
- [ ] Medição de tempo com `cudaEvent`
- [ ] Usar múltiplos blocos (grid maior)
