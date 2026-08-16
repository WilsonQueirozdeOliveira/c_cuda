# Makefile inteligente - detecta automaticamente a arquitetura da GPU
# Funciona no Windows (PowerShell/CMD) e no Linux/WSL

NVCC ?= nvcc

# Detecta Compute Capability de forma portátil (Windows + Linux)
# nvidia-smi retorna algo como "7.5" → vira "75"
ifeq ($(OS),Windows_NT)
  # No Windows: evita head/tr (não existem no cmd)
  RAW_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>NUL)
else
  RAW_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
endif

# Remove espaços e o ponto: "7.5" → "75", "5.0" → "50"
COMPUTE_CAP := $(subst .,,$(strip $(RAW_CAP)))

# Fallback se a detecção falhar
ifeq ($(COMPUTE_CAP),)
  COMPUTE_CAP := 75
endif

# Permite override manual: make ARCH=sm_50
ARCH ?= sm_$(COMPUTE_CAP)
NVCC_FLAGS := -arch=$(ARCH) -O2

ifeq ($(OS),Windows_NT)
  EXE := .exe
  RM := del /Q
else
  EXE :=
  RM := rm -f
endif

.PHONY: all clean info

all: info cuda_1$(EXE) cuda_2$(EXE) cuda_3$(EXE) cuda_4$(EXE) cuda_5$(EXE) cuda_6$(EXE) cuda_7$(EXE) cuda_8$(EXE)

info:
	@echo ========================================
	@echo  Arquitetura detectada: $(ARCH)
	@echo ========================================

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

cuda_8$(EXE): cuda_8.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS)

clean:
ifeq ($(OS),Windows_NT)
	-$(RM) cuda_1.exe cuda_2.exe cuda_3.exe cuda_4.exe cuda_5.exe cuda_6.exe cuda_7.exe cuda_8.exe 2>NUL
else
	$(RM) cuda_1 cuda_2 cuda_3 cuda_4 cuda_5 cuda_6 cuda_7 cuda_8
endif
