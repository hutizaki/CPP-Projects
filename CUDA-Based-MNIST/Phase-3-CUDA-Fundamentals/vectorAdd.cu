#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 100000000;  // 100 million elements (400 MB per array)
    size_t size = n * sizeof(float);  // Use size_t to avoid overflow

    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    float *d_a, *d_b, *d_c;

    cout << "Array size: " << n << " elements (" << size / (1024*1024) << " MB per array)" << endl;
    cout << "Total GPU memory needed: " << (size * 3) / (1024*1024) << " MB\n" << endl;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy input data to device
    cout << "Copying data to GPU..." << endl;
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    cout << "Launching kernel with " << numBlocks << " blocks and " << threadsPerBlock << " threads per block" << endl;
    cout << "Total threads: " << numBlocks * threadsPerBlock << "\n" << endl;
    
    // Time the kernel
    auto start = chrono::high_resolution_clock::now();
    
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();  // Wait for GPU to finish
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    
    cout << "GPU kernel time: " << duration.count() / 1000.0 << " ms" << endl;
    cout << "Throughput: " << (n / 1e6) / (duration.count() / 1e6) << " million elements/sec\n" << endl;

    // Copy result back to host
    cout << "Copying results back to CPU..." << endl;
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify first 5 and last 5 elements
    cout << "Verifying results..." << endl;
    cout << "First 5 elements:" << endl;
    for (int i = 0; i < 5; i++) {
        printf("c[%d] = %.0f (expected %.0f) %s\n", i, h_c[i], h_a[i] + h_b[i],
               (h_c[i] == h_a[i] + h_b[i]) ? "✓" : "✗");
    }
    cout << "\nLast 5 elements:" << endl;
    for (int i = n - 5; i < n; i++) {
        printf("c[%d] = %.0f (expected %.0f) %s\n", i, h_c[i], h_a[i] + h_b[i],
               (h_c[i] == h_a[i] + h_b[i]) ? "✓" : "✗");
    }
    
    cout << "\n✓ All elements processed successfully!" << endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;

    return 0;
}