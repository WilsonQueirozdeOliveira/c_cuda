# c_cuda

Repositório de exemplos básicos em **CUDA C/C++**.

## Exemplos

| Arquivo     | Descrição                                      |
|-------------|------------------------------------------------|
| `cuda_1.cu` | Kernel vazio + `printf` (teste mínimo)         |
| `cuda_2.cu` | Soma de dois números na GPU (`2 + 7`)          |

## Compilação

```bash
# Compilar os dois exemplos
make

# Ou individualmente
nvcc -o cuda_1 cuda_1.cu
nvcc -o cuda_2 cuda_2.cu
```

## Execução

```bash
./cuda_1
./cuda_2
```

Saída esperada de `cuda_2`:
```
2 + 7 = 9
```

## Requisitos

- NVIDIA GPU
- CUDA Toolkit instalado (`nvcc` disponível no PATH)

### Instalação no WSL (Ubuntu)

```bash
# Atualize o sistema
sudo apt update && sudo apt upgrade -y

# Instale o toolkit (versão mais recente recomendada)
sudo apt install nvidia-cuda-toolkit

# Verifique a instalação
nvcc --version
nvidia-smi
```

> **Nota:** No WSL2 é necessário ter o driver NVIDIA instalado no Windows (não dentro do WSL).

### Instalação no Windows (nativo)

1. Baixe o [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) no site da NVIDIA
2. Instale normalmente
3. Abra um novo terminal e verifique:

```powershell
nvcc --version
```

## Estrutura do Projeto

```
c_cuda/
├── cuda_1.cu      # Exemplo 1 - Kernel vazio
├── cuda_2.cu      # Exemplo 2 - Soma na GPU
├── makefile       # Compilação automática
└── README.md
```

## Próximos passos (sugestões)

- [ ] Adicionar tratamento de erros CUDA (`cudaGetLastError`)
- [ ] Exemplo de soma de vetores
- [ ] Exemplo de multiplicação de matrizes
- [ ] Medição de tempo com `cudaEvent`
