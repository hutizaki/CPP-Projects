#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../bryan-library/matrixMath.h"

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void matVecMulKernel(float* matrix, float* vector, float* result, int rows, int cols) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < rows) {
        float sum = 0.0f;

        for (int col = 0; col < cols; col++) {
            sum += matrix[row * cols + col] * vector[col];
        }

        result[row] = sum;
    }
}

void printArr(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        if (i == 0) cout << "[" << arr[i];
        else {
            cout << ", " << arr[i];
        }
    }
    cout << "]" << endl;
}

void testSize(int rows, int cols, bool verbose = false) {
    if (verbose) {
        cout << "\n" << string(60, '=') << endl;
        cout << "Testing: " << rows << " x " << cols << " matrix" << endl;
        cout << "Total elements: " << (rows * cols) << " (" 
             << (rows * cols * sizeof(float)) / (1024.0 * 1024.0) << " MB)" << endl;
        cout << string(60, '=') << endl;
    }
    
    float* h_matrix = new float[rows * cols];
    float* h_vector = new float[cols];
    float* h_result = new float[rows];

    size_t mSize = rows * cols * sizeof(float);
    size_t vSize = cols * sizeof(float);
    size_t rSize = rows * sizeof(float);

    for (int i = 0; i < rows * cols; i++) {
        h_matrix[i] = i + 1;
        if (i < cols) h_vector[i] = i + 1;
    }

    // ========== CPU COMPUTATION ==========
    if (verbose) cout << "\nCPU COMPUTATION..." << endl;
    
    vector<vector<float>> cpu_matrix(rows, vector<float>(cols));
    vector<float> cpu_vector(cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cpu_matrix[i][j] = h_matrix[i * cols + j];
        }
    }
    
    for (int i = 0; i < cols; i++) {
        cpu_vector[i] = h_vector[i];
    }
    
    auto cpu_start = chrono::high_resolution_clock::now();
    vector<float> cpu_result = mm::matrixVectorMultiplication(cpu_matrix, cpu_vector);
    auto cpu_end = chrono::high_resolution_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start);
    
    double cpu_time_ms = cpu_duration.count() / 1000.0;
    if (verbose) cout << "CPU time: " << cpu_time_ms << " ms" << endl;
    
    // ========== GPU COMPUTATION ==========
    if (verbose) cout << "\nGPU COMPUTATION..." << endl;
    
    float *d_matrix, *d_vector, *d_result;

    // Time full GPU operation including memory transfers
    auto gpu_total_start = chrono::high_resolution_clock::now();
    
    CUDA_CHECK(cudaMalloc(&d_matrix, mSize));
    CUDA_CHECK(cudaMalloc(&d_vector, vSize));
    CUDA_CHECK(cudaMalloc(&d_result, rSize));

    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, mSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vector, h_vector, vSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int numBlocks = (rows + threadsPerBlock - 1) / threadsPerBlock;

    auto gpu_kernel_start = chrono::high_resolution_clock::now();
    matVecMulKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_vector, d_result, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_kernel_end = chrono::high_resolution_clock::now();
    
    CUDA_CHECK(cudaMemcpy(h_result, d_result, rSize, cudaMemcpyDeviceToHost));
    
    auto gpu_total_end = chrono::high_resolution_clock::now();
    auto gpu_kernel_duration = chrono::duration_cast<chrono::microseconds>(gpu_kernel_end - gpu_kernel_start);
    auto gpu_total_duration = chrono::duration_cast<chrono::microseconds>(gpu_total_end - gpu_total_start);
    
    double gpu_kernel_time_ms = gpu_kernel_duration.count() / 1000.0;
    double gpu_total_time_ms = gpu_total_duration.count() / 1000.0;
    
    if (verbose) {
        cout << "GPU kernel time: " << gpu_kernel_time_ms << " ms" << endl;
        cout << "GPU total time (with transfers): " << gpu_total_time_ms << " ms" << endl;
    }
    
    // ========== RESULTS ==========
    double speedup_kernel = cpu_time_ms / gpu_kernel_time_ms;
    double speedup_total = cpu_time_ms / gpu_total_time_ms;
    
    printf("%-8d x %-8d | CPU: %8.3f ms | GPU kernel: %8.3f ms | GPU total: %8.3f ms | Speedup (kernel): %6.2fx | Speedup (total): %6.2fx | %s\n",
           rows, cols, cpu_time_ms, gpu_kernel_time_ms, gpu_total_time_ms, 
           speedup_kernel, speedup_total, 
           (speedup_total > 1.0) ? "✓ GPU wins" : "✗ CPU wins");
    
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_vector));
    CUDA_CHECK(cudaFree(d_result));
    
    delete[] h_matrix;
    delete[] h_vector;
    delete[] h_result;
}

int main() {
    // Test with a single size first (original)
    cout << "========== SINGLE SIZE TEST ==========" << endl;
    testSize(300, 400, true);
    
    // Now test multiple sizes to find break-even point
    cout << "\n\n========== MULTI-SIZE BENCHMARK ==========" << endl;
    cout << "Finding break-even point where GPU becomes faster...\n" << endl;
    printf("%-8s x %-8s | %-12s | %-18s | %-18s | %-20s | %-20s | %s\n",
           "Rows", "Cols", "CPU (ms)", "GPU kernel (ms)", "GPU total (ms)", 
           "Speedup (kernel)", "Speedup (total)", "Winner");
    cout << string(140, '-') << endl;
    
    // Test progressively larger sizes
    int test_sizes[][2] = {
        {100, 200},
        {300, 400},
        {500, 600},
        {1000, 1000},
        {2000, 2000},
        {3000, 3000},
        {5000, 5000},
        {10000, 10000}
    };
    
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_tests; i++) {
        int rows = test_sizes[i][0];
        int cols = test_sizes[i][1];
        
        // Skip if too large (memory constraints)
        size_t total_mb = (rows * cols * sizeof(float) * 2 + rows * sizeof(float) + cols * sizeof(float)) / (1024 * 1024);
        if (total_mb > 2000) {  // Skip if > 2GB
            cout << "Skipping " << rows << "x" << cols << " (would need " << total_mb << " MB)" << endl;
            continue;
        }
        
        testSize(rows, cols, false);
        
        // Check if we've found break-even (GPU total faster than CPU)
        // We'll check this in the output, but for now just test all sizes
    }
    
    cout << "\n" << string(140, '-') << endl;
    cout << "\nKey Insights:" << endl;
    cout << "1. GPU kernel time is often faster, but memory transfers add overhead" << endl;
    cout << "2. For small matrices (< 1000x1000), CPU is usually faster due to transfer overhead" << endl;
    cout << "3. For large matrices (> 5000x5000), GPU typically wins significantly" << endl;
    cout << "4. The break-even point depends on your specific GPU and CPU" << endl;
    
    return 0;
}
