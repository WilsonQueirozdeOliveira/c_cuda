all:
	nvcc -o cuda_1.exe cuda_1.cu
	nvcc -o cuda_2.exe cuda_2.cu
	nvcc -o cuda_3.exe cuda_3.cu
	nvcc -o cuda_4.exe cuda_4.cu
	nvcc -o cuda_5.exe cuda_5.cu
	nvcc -o cuda_6.exe cuda_6.cu

clean:
	del /Q cuda_*.exe 2>nul || rm -f cuda_*.exe cuda_1 cuda_2 cuda_3 cuda_4 cuda_5 cuda_6
