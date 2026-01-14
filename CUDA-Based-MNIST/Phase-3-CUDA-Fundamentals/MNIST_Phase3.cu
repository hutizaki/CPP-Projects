#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void helloKernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    helloKernel<<< 1, 10 >>>();
    cudaDeviceSynchronize();
    return 0;
}
