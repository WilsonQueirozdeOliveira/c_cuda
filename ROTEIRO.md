# Roteiro de Aulas – CUDA C/C++

Roteiro progressivo de aprendizado em CUDA, do básico ao intermediário/avançado.

Baseado nos exemplos do repositório `c_cuda` + próximos passos recomendados.

---

## Módulo 1 – Fundamentos

| Aula | Tema | Exemplo no repo |
|------|------|-----------------|
| 1 | O que é CUDA e modelo de execução (Host vs Device) | Conceito |
| 2 | Primeiro kernel + sintaxe `<<<grid, block>>>` | `cuda_1.cu` |
| 3 | Alocação de memória (`cudaMalloc`) e cópia (`cudaMemcpy`) | `cuda_2.cu` |
| 4 | Tratamento de erros CUDA | `cuda_2.cu`, `cuda_3.cu` |
| 5 | Indexação de threads (1D) | `cuda_3.cu` |

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

| Aula | Tema | Exemplo no repo |
|------|------|-----------------|
| 6 | Soma de vetores (paralelismo real) | `cuda_3.cu` |
| 7 | Grid e Block em 2D (`dim3`) | `cuda_4.cu` |
| 8 | Multiplicação de matrizes (Global Memory) | `cuda_4.cu` |
| 9 | Medição de tempo com `cudaEvent` | `cuda_4` a `cuda_6` |
| 10 | Hierarquia de memória CUDA (Global, Shared, Local, Registers, Constant) | Conceito |

**Objetivo do módulo:** Entender como mapear problemas 1D/2D em threads e medir performance.

---

### Aula 10 – Hierarquia de Memória CUDA

A GPU tem vários tipos de memória, cada um com velocidade, tamanho e escopo diferentes. Entender isso é fundamental para otimizar programas.

#### Visão geral (do mais rápido para o mais lento)

| Tipo de Memória | Escopo | Velocidade | Tamanho típico | Quem acessa |
|-----------------|--------|------------|----------------|-------------|
| **Registers** | 1 thread | Mais rápida | Muito pequeno | Só a própria thread |
| **Local Memory** | 1 thread | Lenta* | Automática | Só a própria thread |
| **Shared Memory** | 1 block | Muito rápida | ~48–164 KB / SM | Todas as threads do block |
| **Constant Memory** | Toda a GPU | Rápida (cache) | 64 KB | Todas as threads (leitura) |
| **Global Memory** | Toda a GPU | Lenta | GBs (2–24 GB+) | Todas as threads |

\* Local Memory na verdade fica na Global Memory (é lenta). Só é usada quando faltam registradores.

---

#### 1. Registers (Registradores)

- Mais rápida de todas
- Cada thread tem seus próprios
- Usada para variáveis locais do kernel (`int i`, `float sum`, etc.)
- Limitada: se usar demais, o compilador “derrama” para Local Memory (fica lento)

```cuda
__global__ void exemplo() {
    int idx = threadIdx.x;   // fica em registrador
    float temp = 3.14f;      // fica em registrador
}
```

---

#### 2. Local Memory

- Também privada por thread
- Usada automaticamente quando:
  - Arrays grandes declarados no kernel
  - Muitos registradores necessários (register spilling)
- Na prática é **lenta** (fica na Global Memory)

---

#### 3. Shared Memory (`__shared__`)

- Rápida (fica dentro do SM)
- **Compartilhada por todas as threads do mesmo block**
- Ideal para colaboração entre threads (reduction, tiled matrix mul, etc.)
- Usada nos exemplos `cuda_5`, `cuda_6` e `cuda_7`

```cuda
__shared__ float sdata[256];   // visível para todo o block
```

---

#### 4. Constant Memory (`__constant__`)

- Somente leitura
- Cache especial (rápida quando todas as threads leem o mesmo endereço)
- Boa para coeficientes, tabelas pequenas, parâmetros que não mudam

```cuda
__constant__ float coeficientes[64];
```

---

#### 5. Global Memory

- Maior de todas (vários GB)
- Acessível por todas as threads de todos os blocks
- **Mais lenta**
- É onde ficam os dados principais (`cudaMalloc`)
- Acesso deve ser **coalescido** (threads vizinhas acessando endereços vizinhos) para ser eficiente

```cuda
float *d_data;
cudaMalloc(&d_data, N * sizeof(float));  // Global Memory
```

---

#### Resumo visual da hierarquia

```
Registradores     →  por thread          (mais rápido)
Shared Memory     →  por block           (muito rápido)
Constant Memory   →  toda GPU (leitura)  (rápido com cache)
Global Memory     →  toda GPU            (lento, mas grande)
```

---

#### Regra prática de otimização

1. Dados reutilizados por várias threads do mesmo block → **Shared Memory**
2. Dados que todas as threads leem (e não mudam) → **Constant Memory**
3. Dados grandes e principais → **Global Memory** (acesse de forma coalescida)
4. Variáveis simples de cada thread → **Registers**

---

## Módulo 3 – Shared Memory e Otimização

| Aula | Tema | Exemplo no repo |
|------|------|-----------------|
| 11 | Shared Memory – conceitos e por que usar | Conceito |
| 12 | Tiled Matrix Multiplication | `cuda_5.cu` |
| 13 | Comparação Global Memory vs Shared Memory | `cuda_6.cu` |
| 14 | Bank Conflicts | Próximo |
| 15 | Occupancy e limites do hardware | Próximo |

**Objetivo do módulo:** Aprender a usar Shared Memory para reduzir acessos à memória global e ganhar performance.

---

## Módulo 4 – Padrões Avançados de Algoritmos

| Aula | Tema | Exemplo no repo |
|------|------|-----------------|
| 16 | Parallel Reduction | `cuda_7.cu` |
| 17 | Reduction otimizada (warp-level + `__shfl`) | Próximo |
| 18 | Scan / Prefix Sum | Próximo |
| 19 | Histogramas | Próximo |
| 20 | Stencil e convoluções simples | Próximo |

**Objetivo do módulo:** Dominar padrões clássicos usados em bibliotecas e aplicações reais.

---

## Módulo 5 – Streams, Concorrência e Multi-GPU

| Aula | Tema | Exemplo no repo |
|------|------|-----------------|
| 21 | CUDA Streams básicos | `cuda_8.cu` |
| 22 | Overlap de transferência + execução de kernel | `cuda_8.cu` |
| 23 | Eventos e sincronização avançada | Próximo |
| 24 | Multi-GPU básico | Próximo |
| 25 | Introdução a CUDA Graphs | Avançado |

**Objetivo do módulo:** Esconder latência de transferência e explorar concorrência.

---

## Módulo 6 – Tópicos Modernos e Ferramentas

| Aula | Tema | Observação |
|------|------|------------|
| 26 | Unified Memory | Mais simples, menos controle |
| 27 | Cooperative Groups | API moderna |
| 28 | Warp-level primitives (`__shfl`, `__ballot`, etc.) | Performance |
| 29 | Integração com bibliotecas (cuBLAS, Thrust, cuDNN) | Prático |
| 30 | Profiling com Nsight Compute e Nsight Systems | Essencial |

**Objetivo do módulo:** Saber usar ferramentas profissionais e bibliotecas de alto desempenho.

---

## Progresso Atual do Repositório

- [x] Módulo 1 – Fundamentos
- [x] Módulo 2 – Memória e Performance Básica
- [x] Módulo 3 – Shared Memory (parcial – falta Bank Conflicts e Occupancy)
- [x] Módulo 4 – Parallel Reduction
- [x] Módulo 5 – Streams e Overlap (básico)
- [ ] Módulo 6 – Tópicos modernos

---

## Próximos exemplos sugeridos

1. **cuda_9** – Bank Conflicts (demonstração + versão corrigida)
2. **cuda_10** – Reduction warp-level (otimizada)
3. **cuda_11** – Prefix Sum (Scan)
4. **cuda_12** – Uso básico de cuBLAS
5. Profiling com Nsight Systems

---

## Máquinas usadas neste repositório

| GPU | Compute Capability | Ambiente |
|-----|--------------------|----------|
| GTX 1660 | 7.5 (Turing) | Windows |
| GeForce MX130 | 5.0 (Maxwell) | WSL2 Ubuntu 22.04 |

---

## Como estudar

1. Leia o código do exemplo
2. Compile e rode (`make` + `./cuda_X`)
3. Modifique parâmetros (tamanho, block size, tile size)
4. Meça o impacto no tempo
5. Só então passe para o próximo conceito
