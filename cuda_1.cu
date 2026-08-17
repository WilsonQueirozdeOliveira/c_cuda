#include <stdio.h>

__global__ void kernel() {}

int main() {
    printf("cuda\n");
    kernel<<<1,1>>>();
    printf("fim\n");
    return 0;
}
