all:
	nvcc -o cuda_1.exe cuda_1.cu
	nvcc -o cuda_2.exe cuda_2.cu

clean:
	del /Q cuda_1.exe cuda_2.exe 2>nul || rm -f cuda_1.exe cuda_2.exe cuda_1 cuda_2
