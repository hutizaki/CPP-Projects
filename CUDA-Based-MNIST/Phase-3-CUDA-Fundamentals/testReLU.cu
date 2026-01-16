#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__global__ void reluKernel(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        output[idx] = fmaxf(0, input[idx]);
    }
}

__global__ void sigmoidKernel(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        output[idx] = 1 / (1 + expf(-idx));
    }
}

/*
Steps for kernel coding:
    - Make input and output pointers
    x Create byteSize variable to get number of bytes for m-alloc
    - Allocate memory on the gpu
    - Pass the input to the gpu (copy), with return pointers
    - Grab data from the gpu and copy to pc
    - Delete the pointers for input
    - Free memory from gpu
*/

int main() {
    int n = 101;
    size_t byteSize = n * sizeof(float);

    float* h_arr = new float[n];
    float* h_res = new float[n];

    cout << "Input Array: [";
    for (int i = 0; i < n ; i++) {
        h_arr[i] = i - (n/2);
        if (i == 0) {
            cout << h_arr[i];
        } else {
            cout << ", " << h_arr[i];
        }
    }

    cout << "]" << endl;

    float *d_arr, *d_res;

    cudaMalloc(&d_arr, byteSize);
    cudaMalloc(&d_res, byteSize);

    cudaMemcpy(d_arr, h_arr, byteSize, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 16;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    sigmoidKernel<<<numBlocks, threadsPerBlock>>>(d_arr, d_res, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_res, d_res, byteSize, cudaMemcpyDeviceToHost);


    cout << "Output Array: [";
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            cout << h_res[i];
        } else {
            cout << ", " << h_res[i];
        }
    }

    cout << "]" << endl;

    delete[] h_arr; delete[] h_res;
    cudaFree(d_arr); cudaFree(d_res);
    return 0;
}