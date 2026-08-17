# Roteiro de Aulas – CUDA C/C++

Roteiro progressivo de aprendizado em CUDA, do básico ao intermediário/avançado.

Baseado nos exemplos do repositório `c_cuda` + próximos passos recomendados.

---

## Módulo 1 – Fundamentos

| Aula | Tema | Exemplo no repo | Status |
|------|------|-----------------|--------|
| 1 | O que é CUDA e modelo de execução (Host vs Device) | Conceito | Feito |
| 2 | Primeiro kernel + sintaxe `<<<grid, block>>>` | `cuda_1.cu` | Feito |
| 3 | Alocação de memória (`cudaMalloc`) e cópia (`cudaMemcpy`) | `cuda_2.cu` | Feito |
| 4 | Tratamento de erros CUDA | `cuda_2.cu`, `cuda_3.cu` | Feito |
| 5 | Indexação de threads (1D) | `cuda_3.cu` | Feito |

**Objetivo do módulo:** Conseguir escrever, compilar e executar um kernel simples com verificação de erros.

---

### Aula 1 – O que é CUDA e o Modelo de Execução

#### O que é CUDA?

**CUDA** (Compute Unified Device Architecture) é a plataforma da NVIDIA que permite usar a **GPU** para fazer cálculos gerais (não só gráficos).

Em vez de a GPU só desenhar imagens, você pode usá-la para:
- Somar vetores
- Multiplicar matrizes
- Treinar redes neurais
- Simulações científicas
- Processamento de imagens, etc.

A ideia principal: **a GPU tem milhares de núcleos simples** que podem trabalhar ao mesmo tempo. Ideal para problemas **paralelos**.

---

#### Host vs Device

| Nome | O que é | Quem é |
|------|---------|--------|
| **Host** | CPU + memória RAM | Seu processador normal |
| **Device** | GPU + memória da GPU | Placa de vídeo NVIDIA |

- O **código normal** (C/C++) roda no **Host** (CPU).
- O **código paralelo** (kernel) roda no **Device** (GPU).

Você precisa:
1. Preparar os dados na CPU
2. Copiar para a GPU
3. Executar o kernel na GPU
4. Copiar o resultado de volta para a CPU

---

#### Modelo de Execução CUDA

Quando você lança um kernel assim:

```cuda
kernel<<<grid, block>>>(...);
```

A CUDA organiza as threads em uma hierarquia:

```
Grid
 └── Block 0
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 └── Block 1
 │    ├── Thread 0
 │    └── ...
 └── Block N
```

#### Conceitos principais:

| Termo | Significado |
|-------|-------------|
| **Thread** | Uma unidade de execução (faz um pedaço do trabalho) |
| **Block** | Grupo de threads que colaboram entre si (compartilham Shared Memory) |
| **Grid** | Conjunto de todos os blocks |
| **Kernel** | A função que roda na GPU (`__global__`) |

---

#### Exemplo visual simples

Imagine somar dois vetores de 1024 elementos:

```cuda
add<<<4, 256>>>(a, b, c);   // 4 blocks × 256 threads = 1024 threads
```

- Cada **thread** calcula **um** elemento: `c[i] = a[i] + b[i]`
- As 1024 threads rodam **ao mesmo tempo** (ou quase)

---

#### Fluxo típico de um programa CUDA

```
1. Alocar memória na GPU          →  cudaMalloc
2. Copiar dados CPU → GPU         →  cudaMemcpy (HostToDevice)
3. Lançar o kernel                →  kernel<<<grid, block>>>()
4. Esperar a GPU terminar         →  cudaDeviceSynchronize()
5. Copiar resultado GPU → CPU     →  cudaMemcpy (DeviceToHost)
6. Liberar memória                →  cudaFree
```

---

#### Resumo em uma frase

> **CUDA permite que você escreva funções (kernels) que são executadas por milhares de threads em paralelo na GPU, organizadas em blocks e grids, enquanto a CPU controla o fluxo principal do programa.**

---

## Módulo 2 – Memória e Performance Básica

| Aula | Tema | Exemplo no repo | Status |
|------|------|-----------------|--------|
| 6 | Soma de vetores (paralelismo real) | `cuda_3.cu` | Feito |
| 7 | Grid e Block em 2D (`dim3`) | `cuda_4.cu` | Feito |
| 8 | Multiplicação de matrizes (Global Memory) | `cuda_4.cu` | Feito |
| 9 | Medição de tempo com `cudaEvent` | `cuda_4` a `cuda_6` | Feito |
| 10 | Hierarquia de memória CUDA | Conceito | Feito |

**Objetivo do módulo:** Entender como mapear problemas 1D/2D em threads e medir performance.

---

### Aula 10 – Hierarquia de Memória CUDA

A GPU tem vários tipos de memória, cada um com velocidade, tamanho e escopo diferentes.

#### Visão geral (do mais rápido para o mais lento)

| Tipo de Memória | Escopo | Velocidade | Tamanho típico | Quem acessa |
|-----------------|--------|------------|----------------|-------------|
| **Registers** | 1 thread | Mais rápida | Muito pequeno | Só a própria thread |
| **Local Memory** | 1 thread | Lenta* | Automática | Só a própria thread |
| **Shared Memory** | 1 block | Muito rápida | ~48–164 KB / SM | Todas as threads do block |
| **Constant Memory** | Toda a GPU | Rápida (cache) | 64 KB | Todas as threads (leitura) |
| **Global Memory** | Toda a GPU | Lenta | GBs (2–24 GB+) | Todas as threads |

\* Local Memory na verdade fica na Global Memory (é lenta). Só é usada quando faltam registradores.

#### Regra prática de otimização

1. Dados reutilizados por várias threads do mesmo block → **Shared Memory**
2. Dados que todas as threads leem (e não mudam) → **Constant Memory**
3. Dados grandes e principais → **Global Memory** (acesse de forma coalescida)
4. Variáveis simples de cada thread → **Registers**

---

## Módulo 3 – Shared Memory e Otimização

| Aula | Tema | Exemplo no repo | Status |
|------|------|-----------------|--------|
| 11 | Shared Memory – conceitos e por que usar | Conceito | Feito |
| 12 | Tiled Matrix Multiplication | `cuda_5.cu` | Feito |
| 13 | Comparação Global Memory vs Shared Memory | `cuda_6.cu` | Feito |
| 14 | Bank Conflicts | Próximo | Pendente |
| 15 | Occupancy e limites do hardware | Próximo | Pendente |

**Objetivo do módulo:** Aprender a usar Shared Memory para reduzir acessos à memória global e ganhar performance.

---

### Aula 11 – Shared Memory – conceitos e por que usar

#### O que é Shared Memory?

É uma memória **rápida** e **pequena** que fica **dentro do Streaming Multiprocessor (SM)**.

- Todas as threads do **mesmo block** conseguem ler e escrever nela
- Threads de blocks diferentes **não** veem a Shared Memory uma da outra
- É cerca de **100× mais rápida** que a Global Memory

```cuda
__shared__ float sdata[256];   // declarada dentro do kernel
```

#### Por que ela existe?

A **Global Memory** é grande, mas lenta.  
A **Shared Memory** serve como uma **cache manual** controlada pelo programador.

Você carrega os dados da Global → Shared **uma vez**, e depois as threads reutilizam esses dados muitas vezes sem voltar na memória lenta.

#### Analogia simples

| Tipo de memória | Analogia |
|-----------------|----------|
| Global Memory | Estoque no porão (longe e lento) |
| Shared Memory | Mesa de estudo do grupo (perto e rápida) |
| Registers | Caderno pessoal de cada aluno |

#### Quando usar?

- Várias threads do mesmo block precisam **ler os mesmos dados**
- Você vai **reutilizar** os mesmos valores várias vezes
- Precisa de **colaboração** entre threads (reduction, scan, matrix mul tiled, etc.)

Exemplos no repositório: `cuda_5`, `cuda_6`, `cuda_7`.

#### Ciclo típico de uso

```cuda
__global__ void exemplo(float *input, float *output) {
    __shared__ float sdata[256];

    // 1. Carregar da Global → Shared
    sdata[threadIdx.x] = input[...];
    __syncthreads();                 // espera todo mundo carregar

    // 2. Processar usando só Shared Memory (rápido)
    // ...

    __syncthreads();

    // 3. Escrever resultado de volta na Global (se necessário)
    if (threadIdx.x == 0)
        output[blockIdx.x] = sdata[0];
}
```

O `__syncthreads()` é **obrigatório** para garantir que todas as threads terminaram de carregar (ou de usar) os dados antes de continuar.

#### Resumo

> **Shared Memory é a ferramenta principal de otimização em CUDA.**  
> Ela permite que as threads de um block trabalhem juntas em cima de dados rápidos, reduzindo drasticamente o número de acessos lentos à Global Memory.

No `cuda_6`, a versão com Shared Memory ficou **~2× mais rápida** (MX130 e GTX 1660).

---

## Módulo 4 – Padrões Avançados de Algoritmos

| Aula | Tema | Exemplo no repo | Status |
|------|------|-----------------|--------|
| 16 | Parallel Reduction | `cuda_7.cu` | Feito |
| 17 | Reduction otimizada (warp-level + `__shfl`) | Próximo | Pendente |
| 18 | Scan / Prefix Sum | Próximo | Pendente |
| 19 | Histogramas | Próximo | Pendente |
| 20 | Stencil e convoluções simples | Próximo | Pendente |

**Objetivo do módulo:** Dominar padrões clássicos usados em bibliotecas e aplicações reais.

---

## Módulo 5 – Streams, Concorrência e Multi-GPU

| Aula | Tema | Exemplo no repo | Status |
|------|------|-----------------|--------|
| 21 | CUDA Streams básicos | `cuda_8.cu` | Feito |
| 22 | Overlap de transferência + execução de kernel | `cuda_8.cu` | Feito |
| 23 | Eventos e sincronização avançada | Próximo | Pendente |
| 24 | Multi-GPU básico | Próximo | Pendente |
| 25 | Introdução a CUDA Graphs | Avançado | Pendente |

**Objetivo do módulo:** Esconder latência de transferência e explorar concorrência.

---

## Módulo 6 – Tópicos Modernos e Ferramentas

| Aula | Tema | Observação | Status |
|------|------|------------|--------|
| 26 | Unified Memory | Mais simples, menos controle | Pendente |
| 27 | Cooperative Groups | API moderna | Pendente |
| 28 | Warp-level primitives (`__shfl`, `__ballot`, etc.) | Performance | Pendente |
| 29 | Integração com bibliotecas (cuBLAS, Thrust, cuDNN) | Prático | Pendente |
| 30 | Profiling com Nsight Compute e Nsight Systems | Essencial | Pendente |

**Objetivo do módulo:** Saber usar ferramentas profissionais e bibliotecas de alto desempenho.

---

## Progresso Atual do Repositório

### Concluído
- [x] Módulo 1 – Fundamentos (aulas 1–5)
- [x] Módulo 2 – Memória e Performance Básica (aulas 6–10)
- [x] Módulo 3 – Shared Memory (aulas 11–13; faltam 14–15)
- [x] Módulo 4 – Parallel Reduction (aula 16)
- [x] Módulo 5 – Streams e Overlap (aulas 21–22)
- [x] Makefile inteligente (Windows + Linux/WSL)
- [x] Benchmark MX130 vs GTX 1660
- [x] Comparação WSL vs Windows nativo

### Pendente
- [ ] Bank Conflicts e Occupancy
- [ ] Reduction warp-level, Scan, Histogramas, Stencil
- [ ] Multi-GPU e CUDA Graphs
- [ ] Unified Memory, Cooperative Groups, cuBLAS
- [ ] Profiling com Nsight

---

## Próximos exemplos sugeridos

1. **cuda_9** – Bank Conflicts (demonstração + versão corrigida)
2. **cuda_10** – Reduction warp-level (otimizada)
3. **cuda_11** – Prefix Sum (Scan)
4. **cuda_12** – Uso básico de cuBLAS
5. Profiling com Nsight Systems

---

## Máquinas usadas neste repositório

| GPU | Compute Capability | Arquitetura | Ambiente |
|-----|--------------------|-------------|----------|
| GTX 1660 | 7.5 | Turing | Desktop (WSL2 + Windows nativo) |
| GeForce MX130 | 5.0 | Maxwell | Notebook (WSL2 Ubuntu 22.04) |

---

## Como estudar

1. Leia o código do exemplo
2. Compile e rode (`make` + `./cuda_X`)
3. Modifique parâmetros (tamanho, block size, tile size)
4. Meça o impacto no tempo
5. Só então passe para o próximo conceito
