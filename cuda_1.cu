#include <stdio.h>
#include <iostream>

__global__ void kernel(void) {
}

int main(void) {

	printf("cuda\n");
	kernel<<<1,1>>>();
	printf("fim\n");

	return 0;
}
