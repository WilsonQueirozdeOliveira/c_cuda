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
