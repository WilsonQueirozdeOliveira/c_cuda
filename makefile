# Makefile inteligente - detecta automaticamente a arquitetura da GPU
# Funciona no Windows e no Linux/WSL

NVCC ?= nvcc

# Detecta a Compute Capability (ex: 5.0 -> 50, 7.5 -> 75)
COMPUTE_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.' | tr -d ' ')

ifeq ($(COMPUTE_CAP),)
  COMPUTE_CAP := 50
endif

ARCH := sm_$(COMPUTE_CAP)
NVCC_FLAGS := -arch=$(ARCH) -O2

ifeq ($(OS),Windows_NT)
  EXE := .exe
  RM := del /Q
else
  EXE :=
  RM := rm -f
endif

.PHONY: all clean info

all: info cuda_1$(EXE) cuda_2$(EXE) cuda_3$(EXE) cuda_4$(EXE) cuda_5$(EXE) cuda_6$(EXE) cuda_7$(EXE)

info:
	@echo "========================================"
	@echo " Arquitetura detectada: $(ARCH)"
	@echo "========================================"

cuda_1$(EXE): cuda_1.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS)

cuda_2$(EXE): cuda_2.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS)

cuda_3$(EXE): cuda_3.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS)

cuda_4$(EXE): cuda_4.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS)

cuda_5$(EXE): cuda_5.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS)

cuda_6$(EXE): cuda_6.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS)

cuda_7$(EXE): cuda_7.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS)

clean:
	$(RM) cuda_1$(EXE) cuda_2$(EXE) cuda_3$(EXE) cuda_4$(EXE) cuda_5$(EXE) cuda_6$(EXE) cuda_7$(EXE) 2>/dev/null || true
