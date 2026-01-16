#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void printIDs() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Block %d, Thread %d → Global ID %d\n", blockIdx.x, threadIdx.x, idx);
}

int main() {
    printIDs<<<3, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
