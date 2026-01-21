#include "../bryan-library/machineLearning.h"
#include "../bryan-library/fileParsing.h"
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << static_cast<int>(err) << std::endl; \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

using namespace ml;
using namespace std;
using namespace fp;

struct TrainedWeights {
    vector<vector<float>> w1;
    vector<vector<float>> w2;
    vector<float> b1;
    vector<float> b2;
};

// GPU weights structure (simplified, no bryan-library dependency)
struct GPUWeights {
    float* d_w1;
    float* d_w2;
    float* d_b1;
    float* d_b2;
    int w1_rows, w1_cols;
    int w2_rows, w2_cols;
    int b1_size, b2_size;
    
    GPUWeights(const vector<vector<float>>& w1_host, const vector<vector<float>>& w2_host,
               const vector<float>& b1_host, const vector<float>& b2_host) {
        w1_rows = w1_host.size();
        w1_cols = w1_host[0].size();
        w2_rows = w2_host.size();
        w2_cols = w2_host[0].size();
        b1_size = b1_host.size();
        b2_size = b2_host.size();
        
        // Flatten and allocate on GPU
        vector<float> w1_flat(w1_rows * w1_cols);
        for (int i = 0; i < w1_rows; i++) {
            for (int j = 0; j < w1_cols; j++) {
                w1_flat[i * w1_cols + j] = w1_host[i][j];
            }
        }
        
        vector<float> w2_flat(w2_rows * w2_cols);
        for (int i = 0; i < w2_rows; i++) {
            for (int j = 0; j < w2_cols; j++) {
                w2_flat[i * w2_cols + j] = w2_host[i][j];
            }
        }
        
        CUDA_CHECK(cudaMalloc(&d_w1, w1_rows * w1_cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w2, w2_rows * w2_cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b1, b1_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b2, b2_size * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_w1, w1_flat.data(), w1_rows * w1_cols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w2, w2_flat.data(), w2_rows * w2_cols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, b1_host.data(), b1_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, b2_host.data(), b2_size * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    ~GPUWeights() {
        cudaFree(d_w1);
        cudaFree(d_w2);
        cudaFree(d_b1);
        cudaFree(d_b2);
    }
    
    void copyToHost(vector<vector<float>>& w1_host, vector<vector<float>>& w2_host,
                    vector<float>& b1_host, vector<float>& b2_host) {
        vector<float> w1_flat(w1_rows * w1_cols);
        vector<float> w2_flat(w2_rows * w2_cols);
        
        CUDA_CHECK(cudaMemcpy(w1_flat.data(), d_w1, w1_rows * w1_cols * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w2_flat.data(), d_w2, w2_rows * w2_cols * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b1_host.data(), d_b1, b1_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b2_host.data(), d_b2, b2_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        w1_host.resize(w1_rows);
        for (int i = 0; i < w1_rows; i++) {
            w1_host[i].resize(w1_cols);
            for (int j = 0; j < w1_cols; j++) {
                w1_host[i][j] = w1_flat[i * w1_cols + j];
            }
        }
        
        w2_host.resize(w2_rows);
        for (int i = 0; i < w2_rows; i++) {
            w2_host[i].resize(w2_cols);
            for (int j = 0; j < w2_cols; j++) {
                w2_host[i][j] = w2_flat[i * w2_cols + j];
            }
        }
    }
};

// ReLU kernel
__global__ void reluKernel(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
    }
}

// Bias add kernel
__global__ void biasAddKernel(float* input, float* bias, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int idx = threadIdx.x;
    if (batch_idx < batch_size && idx < dim) {
        output[batch_idx * dim + idx] = input[batch_idx * dim + idx] + bias[idx];
    }
}

// Softmax kernel
__global__ void softmaxKernel(float* input, float* output, int batch_size, int n) {
    int batch_idx = blockIdx.x;
    int idx = threadIdx.x;
    
    if (batch_idx < batch_size && idx < n) {
        // Find max for numerical stability
        __shared__ float max_val[256];
        if (threadIdx.x == 0) {
            max_val[0] = input[batch_idx * n];
            for (int i = 1; i < n; i++) {
                if (input[batch_idx * n + i] > max_val[0]) {
                    max_val[0] = input[batch_idx * n + i];
                }
            }
        }
        __syncthreads();
        
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += expf(input[batch_idx * n + i] - max_val[0]);
        }
        
        output[batch_idx * n + idx] = expf(input[batch_idx * n + idx] - max_val[0]) / sum;
    }
}

// Output error kernel (for backward pass)
__global__ void computeOutputErrorKernel(float* y_hat, int* trueLabels, float* outputError,
                                        int batch_size, int n) {
    int batch_idx = blockIdx.x;
    int idx = threadIdx.x;
    
    if (batch_idx < batch_size && idx < n) {
        if (idx == trueLabels[batch_idx]) {
            outputError[batch_idx * n + idx] = y_hat[batch_idx * n + idx] - 1.0f;
        } else {
            outputError[batch_idx * n + idx] = y_hat[batch_idx * n + idx];
        }
    }
}

// ReLU backward kernel
__global__ void reluBackwardKernel(float* activation, float* error, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (activation[idx] <= 0.0f) {
            error[idx] = 0.0f;
        }
    }
}

// Bias gradient kernel
__global__ void biasGradientKernel(float* error, float* gradient, int batch_size, int dim) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < dim) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += error[i * dim + idx];
        }
        gradient[idx] = sum;
    }
}

// Weight update kernel
__global__ void updateWeightsKernel(float* weights, float* gradients, float learning_rate,
                                   int batch_size, int total_elements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total_elements) {
        weights[idx] -= learning_rate * (gradients[idx] / batch_size);
    }
}

// Forward pass using cuBLAS
void batchedForwardPass(cublasHandle_t handle, GPUWeights& weights, float* d_batch_input, 
                        int batch_size, float* d_hiddenZ, float* d_hiddenA, float* d_outputZ) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Layer 1: input -> hidden
    // We want: hiddenZ = batch_input * W1^T
    // Our format (row-major): batch_input [batch_size x input_dim], W1 [hidden_dim x input_dim]
    // cuBLAS uses column-major, so for row-major data:
    // - We compute: C^T = (W1^T)^T * batch_input^T = W1 * batch_input^T
    // - Which gives us: hiddenZ^T = W1 * batch_input^T
    // - So: cublasSgemm(..., CUBLAS_OP_T, CUBLAS_OP_N, ...) with swapped dimensions
    // Actually simpler: use CUBLAS_OP_N, CUBLAS_OP_T with correct leading dims
    // For row-major A [m x k], cuBLAS sees it as column-major [k x m], so lda = m
    // For row-major B [n x k], cuBLAS sees it as column-major [k x n], so ldb = n
    // For row-major C [m x n], cuBLAS sees it as column-major [n x m], so ldc = m
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                              batch_size, weights.w1_rows, weights.w1_cols,
                              &alpha, d_batch_input, batch_size,        // lda = batch_size (rows in row-major)
                              weights.d_w1, weights.w1_rows,           // ldb = hidden_dim (rows in row-major)
                              &beta, d_hiddenZ, batch_size));           // ldc = batch_size (rows in row-major)
    
    // Add bias
    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    biasAddKernel<<<numBlocks, weights.b1_size>>>(d_hiddenZ, weights.d_b1, d_hiddenZ, batch_size, weights.b1_size);
    CUDA_CHECK(cudaGetLastError());
    
    // Apply ReLU
    int total_hidden = batch_size * weights.b1_size;
    numBlocks = (total_hidden + threadsPerBlock - 1) / threadsPerBlock;
    reluKernel<<<numBlocks, threadsPerBlock>>>(d_hiddenZ, d_hiddenA, total_hidden);
    CUDA_CHECK(cudaGetLastError());
    
    // Layer 2: hidden -> output
    // d_outputZ = d_hiddenA * W2^T
    // hiddenA [batch_size x hidden_dim] * W2^T [hidden_dim x output_dim] = outputZ [batch_size x output_dim]
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                              batch_size, weights.w2_rows, weights.w2_cols,
                              &alpha, d_hiddenA, batch_size,        // lda = batch_size
                              weights.d_w2, weights.w2_rows,        // ldb = output_dim
                              &beta, d_outputZ, batch_size));       // ldc = batch_size
    
    // Add bias
    numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    biasAddKernel<<<numBlocks, weights.b2_size>>>(d_outputZ, weights.d_b2, d_outputZ, batch_size, weights.b2_size);
    CUDA_CHECK(cudaGetLastError());
}

// Backward pass using cuBLAS
void batchedBackwardPass(cublasHandle_t handle, GPUWeights& weights,
                         float* d_batch_input, int* d_trueLabels,
                         float* d_hiddenZ, float* d_hiddenA, float* d_outputZ, float* d_y_hat,
                         float* d_dW1, float* d_db1, float* d_dW2, float* d_db2,
                         int batch_size, float learning_rate) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float alpha_grad = 1.0f / batch_size;
    int threadsPerBlock = 256;
    int numBlocks;
    
    // 1. Compute Softmax
    numBlocks = batch_size;
    softmaxKernel<<<numBlocks, weights.b2_size>>>(d_outputZ, d_y_hat, batch_size, weights.b2_size);
    CUDA_CHECK(cudaGetLastError());
    
    // 2. Compute output error (dL/dZ2)
    numBlocks = batch_size;
    computeOutputErrorKernel<<<numBlocks, weights.b2_size>>>(d_y_hat, d_trueLabels, d_outputZ, batch_size, weights.b2_size);
    CUDA_CHECK(cudaGetLastError());
    
    // 3. Compute gradients for W2: dW2 = outputError^T * hiddenA
    // outputError [batch_size x output_dim]^T * hiddenA [batch_size x hidden_dim]
    // = [output_dim x batch_size] * [batch_size x hidden_dim] = [output_dim x hidden_dim]
    // In cuBLAS: C = alpha * op(A) * op(B) + beta * C
    // We want: dW2 = outputError^T * hiddenA
    // So: op(A) = outputError^T, op(B) = hiddenA
    CUDA_CHECK(cudaMemset(d_dW2, 0, weights.w2_rows * weights.w2_cols * sizeof(float)));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              weights.w2_rows, weights.w2_cols, batch_size,
                              &alpha_grad, d_outputZ, batch_size,   // lda = batch_size
                              d_hiddenA, batch_size,                 // ldb = batch_size
                              &beta, d_dW2, weights.w2_rows));      // ldc = output_dim
    
    // 4. Compute gradients for b2
    numBlocks = (weights.b2_size + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK(cudaMemset(d_db2, 0, weights.b2_size * sizeof(float)));
    biasGradientKernel<<<numBlocks, threadsPerBlock>>>(d_outputZ, d_db2, batch_size, weights.b2_size);
    CUDA_CHECK(cudaGetLastError());
    
    // 5. Backpropagate to hidden layer: hiddenError = outputError * W2
    // outputError [batch_size x output_dim] * W2 [output_dim x hidden_dim] = hiddenError [batch_size x hidden_dim]
    // In cuBLAS: C = alpha * A * B + beta * C
    // A = outputError [batch_size x output_dim], lda = output_dim
    // B = W2 [output_dim x hidden_dim], ldb = hidden_dim
    // C = hiddenError [batch_size x hidden_dim], ldc = hidden_dim
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              batch_size, weights.w2_cols, weights.w2_rows,
                              &alpha, d_outputZ, batch_size,        // lda = batch_size
                              weights.d_w2, weights.w2_rows,         // ldb = output_dim
                              &beta, d_hiddenZ, batch_size));        // ldc = batch_size
    
    // 6. Apply ReLU derivative
    int total_hidden = batch_size * weights.b1_size;
    numBlocks = (total_hidden + threadsPerBlock - 1) / threadsPerBlock;
    reluBackwardKernel<<<numBlocks, threadsPerBlock>>>(d_hiddenA, d_hiddenZ, total_hidden);
    CUDA_CHECK(cudaGetLastError());
    
    // 7. Compute gradients for W1: dW1 = hiddenError^T * batch_input
    // hiddenError [batch_size x hidden_dim]^T * batch_input [batch_size x input_dim]
    // = [hidden_dim x batch_size] * [batch_size x input_dim] = [hidden_dim x input_dim]
    CUDA_CHECK(cudaMemset(d_dW1, 0, weights.w1_rows * weights.w1_cols * sizeof(float)));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              weights.w1_rows, weights.w1_cols, batch_size,
                              &alpha_grad, d_hiddenZ, batch_size,   // lda = batch_size
                              d_batch_input, batch_size,              // ldb = batch_size
                              &beta, d_dW1, weights.w1_rows));       // ldc = hidden_dim
    
    // 8. Compute gradients for b1
    numBlocks = (weights.b1_size + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK(cudaMemset(d_db1, 0, weights.b1_size * sizeof(float)));
    biasGradientKernel<<<numBlocks, threadsPerBlock>>>(d_hiddenZ, d_db1, batch_size, weights.b1_size);
    CUDA_CHECK(cudaGetLastError());
    
    // 9. Update weights and biases
    numBlocks = (weights.w2_rows * weights.w2_cols + threadsPerBlock - 1) / threadsPerBlock;
    updateWeightsKernel<<<numBlocks, threadsPerBlock>>>(weights.d_w2, d_dW2, learning_rate, batch_size, weights.w2_rows * weights.w2_cols);
    CUDA_CHECK(cudaGetLastError());
    
    numBlocks = (weights.b2_size + threadsPerBlock - 1) / threadsPerBlock;
    updateWeightsKernel<<<numBlocks, threadsPerBlock>>>(weights.d_b2, d_db2, learning_rate, batch_size, weights.b2_size);
    CUDA_CHECK(cudaGetLastError());
    
    numBlocks = (weights.w1_rows * weights.w1_cols + threadsPerBlock - 1) / threadsPerBlock;
    updateWeightsKernel<<<numBlocks, threadsPerBlock>>>(weights.d_w1, d_dW1, learning_rate, batch_size, weights.w1_rows * weights.w1_cols);
    CUDA_CHECK(cudaGetLastError());
    
    numBlocks = (weights.b1_size + threadsPerBlock - 1) / threadsPerBlock;
    updateWeightsKernel<<<numBlocks, threadsPerBlock>>>(weights.d_b1, d_db1, learning_rate, batch_size, weights.b1_size);
    CUDA_CHECK(cudaGetLastError());
    
    cudaDeviceSynchronize();
}

// Full GPU training using cuBLAS
TrainedWeights runTrainingGPU(const vector<vector<float>> &images, const vector<int> labels, 
                              int numNeurons, int epochs, float learning_rate, int batch_size) {
    int inputLength = images[0].size();
    
    // Try to load initial weights, otherwise generate random
    vector<vector<float>> w1, w2;
    vector<float> b1, b2;
    
    string initial_weights_path = "initial_weights";
    ifstream testFile(initial_weights_path + "_W1.bin", ios::binary);
    if (testFile.is_open()) {
        testFile.close();
        cout << "Loading initial weights for fair comparison..." << endl;
        uint32_t w1_rows, w1_cols;
        ifstream w1File(initial_weights_path + "_W1.bin", ios::binary);
        w1File.read(reinterpret_cast<char*>(&w1_rows), sizeof(w1_rows));
        w1File.read(reinterpret_cast<char*>(&w1_cols), sizeof(w1_cols));
        w1.resize(w1_rows);
        for (auto& row : w1) {
            row.resize(w1_cols);
            w1File.read(reinterpret_cast<char*>(row.data()), w1_cols * sizeof(float));
        }
        w1File.close();
        
        uint32_t w2_rows, w2_cols;
        ifstream w2File(initial_weights_path + "_W2.bin", ios::binary);
        w2File.read(reinterpret_cast<char*>(&w2_rows), sizeof(w2_rows));
        w2File.read(reinterpret_cast<char*>(&w2_cols), sizeof(w2_cols));
        w2.resize(w2_rows);
        for (auto& row : w2) {
            row.resize(w2_cols);
            w2File.read(reinterpret_cast<char*>(row.data()), w2_cols * sizeof(float));
        }
        w2File.close();
        
        uint32_t b1_size;
        ifstream b1File(initial_weights_path + "_b1.bin", ios::binary);
        b1File.read(reinterpret_cast<char*>(&b1_size), sizeof(b1_size));
        b1.resize(b1_size);
        b1File.read(reinterpret_cast<char*>(b1.data()), b1_size * sizeof(float));
        b1File.close();
        
        uint32_t b2_size;
        ifstream b2File(initial_weights_path + "_b2.bin", ios::binary);
        b2File.read(reinterpret_cast<char*>(&b2_size), sizeof(b2_size));
        b2.resize(b2_size);
        b2File.read(reinterpret_cast<char*>(b2.data()), b2_size * sizeof(float));
        b2File.close();
    } else {
        w1 = ml::generateRandMatrix(numNeurons, inputLength);
        w2 = ml::generateRandMatrix(10, numNeurons);
        b1 = ml::generateRandVector(numNeurons);
        b2 = ml::generateRandVector(10);
    }
    
    // Create GPU weights
    GPUWeights gpu_weights(w1, w2, b1, b2);
    
    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    // Enable row-major mode (cuBLAS 11.0+)
    // For older versions, we'll use column-major with transposed operations
    #if CUBLAS_VER_MAJOR >= 11
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    #endif
    
    // Allocate GPU memory
    float *d_batch_input, *d_hiddenZ, *d_hiddenA, *d_outputZ, *d_y_hat;
    int *d_trueLabels;
    float *d_dW1, *d_db1, *d_dW2, *d_db2;
    
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_size * inputLength * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hiddenZ, batch_size * numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hiddenA, batch_size * numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputZ, batch_size * 10 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_hat, batch_size * 10 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trueLabels, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dW1, numNeurons * inputLength * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW2, 10 * numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, 10 * sizeof(float)));
    
    vector<float> batch_y_hat(batch_size * 10);
    
    // Shuffle indices for training
    vector<int> indices(images.size());
    iota(indices.begin(), indices.end(), 0);
    
    // Random number generator for shuffling
    random_device rd;
    mt19937 g(rd());
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        // Shuffle data
        shuffle(indices.begin(), indices.end(), g);
        
        // Process batches
        for (int batch_start = 0; batch_start < images.size(); batch_start += batch_size) {
            int actual_batch_size = min(batch_size, (int)images.size() - batch_start);
            
            // Prepare batch
            vector<float> batch_input(actual_batch_size * inputLength);
            vector<int> batch_labels(actual_batch_size);
            for (int i = 0; i < actual_batch_size; i++) {
                int idx = indices[batch_start + i];
                batch_labels[i] = labels[idx];
                for (int j = 0; j < inputLength; j++) {
                    batch_input[i * inputLength + j] = images[idx][j];
                }
            }
            
            // Transfer to GPU
            CUDA_CHECK(cudaMemcpy(d_batch_input, batch_input.data(), 
                                  actual_batch_size * inputLength * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_trueLabels, batch_labels.data(),
                                  actual_batch_size * sizeof(int),
                                  cudaMemcpyHostToDevice));
            
            // Forward pass
            batchedForwardPass(cublas_handle, gpu_weights, d_batch_input, actual_batch_size,
                              d_hiddenZ, d_hiddenA, d_outputZ);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Backward pass
            batchedBackwardPass(cublas_handle, gpu_weights, d_batch_input, d_trueLabels,
                               d_hiddenZ, d_hiddenA, d_outputZ, d_y_hat,
                               d_dW1, d_db1, d_dW2, d_db2,
                               actual_batch_size, learning_rate);
            
            // Compute loss (copy y_hat back)
            CUDA_CHECK(cudaMemcpy(batch_y_hat.data(), d_y_hat,
                                  actual_batch_size * 10 * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            
            for (int i = 0; i < actual_batch_size; i++) {
                int idx = indices[batch_start + i];
                vector<float> y_hat(10);
                for (int j = 0; j < 10; j++) {
                    y_hat[j] = batch_y_hat[i * 10 + j];
                }
                float loss = categoricalCrossEntropy(y_hat, labels[idx]);
                total_loss += loss;
            }
        }
        
        float avg_loss = total_loss / images.size();
        cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << avg_loss << endl;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_batch_input));
    CUDA_CHECK(cudaFree(d_hiddenZ));
    CUDA_CHECK(cudaFree(d_hiddenA));
    CUDA_CHECK(cudaFree(d_outputZ));
    CUDA_CHECK(cudaFree(d_y_hat));
    CUDA_CHECK(cudaFree(d_trueLabels));
    CUDA_CHECK(cudaFree(d_dW1));
    CUDA_CHECK(cudaFree(d_db1));
    CUDA_CHECK(cudaFree(d_dW2));
    CUDA_CHECK(cudaFree(d_db2));
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    
    // Copy weights back
    gpu_weights.copyToHost(w1, w2, b1, b2);
    
    return {w1, w2, b1, b2};
}

int argmax(const vector<float>& vec) {
    int maxIndex = 0;
    float maxValue = vec[0];
    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] > maxValue) {
            maxValue = vec[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

float testTraining(const vector<vector<float>> &images, 
                   const vector<int> &labels,
                   TrainedWeights &weights) {
    int correct = 0;
    int total = images.size();
    cout << "\nTesting on " << total << " images..." << endl;
    
    for (int i = 0; i < total; i++) {
        vector<float> hiddenZ = processLayer(weights.w1, weights.b1, images[i]);
        vector<float> hiddenA = relu(hiddenZ);
        vector<float> outputZ = processLayer(weights.w2, weights.b2, hiddenA);
        vector<float> y_hat = softmax(outputZ);
        
        int predicted = argmax(y_hat);
        if (predicted == labels[i]) {
            correct++;
        }
    }
    
    float accuracy = (float)correct / total * 100.0f;
    cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "% (" 
         << correct << "/" << total << ")" << endl;
    
    return accuracy;
}

void saveWeights(const string& filename, const TrainedWeights& weights) {
    ofstream w1File(filename + "_W1.bin", ios::binary);
    ofstream w2File(filename + "_W2.bin", ios::binary);
    ofstream b1File(filename + "_b1.bin", ios::binary);
    ofstream b2File(filename + "_b2.bin", ios::binary);
    
    for (const auto& row : weights.w1) {
        w1File.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    for (const auto& row : weights.w2) {
        w2File.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    b1File.write(reinterpret_cast<const char*>(weights.b1.data()), weights.b1.size() * sizeof(float));
    b2File.write(reinterpret_cast<const char*>(weights.b2.data()), weights.b2.size() * sizeof(float));
    
    w1File.close();
    w2File.close();
    b1File.close();
    b2File.close();
    
    cout << "Weights saved to " << filename << "_*.bin" << endl;
}

int main() {
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        cerr << "ERROR: No CUDA devices found or CUDA not available!" << endl;
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "=== CUDA DEVICE INFO ===" << endl;
    cout << "Device: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    cout << endl;
    
    // Load training data
    cout << "=== LOADING TRAINING DATA ===" << endl;
    ifstream trainLabelFile("../train-labels.idx1-ubyte", ios::binary);
    ifstream trainImageFile("../train-images.idx3-ubyte", ios::binary);
    const vector<int> trainLabels = loadMNISTLabels(trainLabelFile);
    const vector<vector<float>> trainImages = loadMNISTImages(trainImageFile);
    trainLabelFile.close();
    trainImageFile.close();
    
    // Training parameters
    int numNeurons = 128;
    int epochs = 10;
    float learning_rate = 0.1f;
    int batch_size = 32;
    
    // === GPU TRAINING (using cuBLAS) ===
    cout << "\n=== GPU TRAINING (Phase 5.5 - cuBLAS) ===" << endl;
    cout << "Using NVIDIA cuBLAS for optimized matrix multiplication!" << endl;
    auto start_gpu = chrono::high_resolution_clock::now();
    TrainedWeights weights_gpu = runTrainingGPU(trainImages, trainLabels, numNeurons, epochs, learning_rate, batch_size);
    auto end_gpu = chrono::high_resolution_clock::now();
    auto duration_gpu = chrono::duration_cast<chrono::milliseconds>(end_gpu - start_gpu);
    
    cout << "\n=== TRAINING TIME ===" << endl;
    cout << "GPU Training Time: " << duration_gpu.count() << " ms (" 
         << fixed << setprecision(2) << duration_gpu.count() / 1000.0 << " seconds)" << endl;
    cout << endl;
    
    // Load test data
    cout << "=== LOADING TEST DATA ===" << endl;
    ifstream testLabelFile("../t10k-labels.idx1-ubyte", ios::binary);
    ifstream testImageFile("../t10k-images.idx3-ubyte", ios::binary);
    const vector<int> testLabels = loadMNISTLabels(testLabelFile);
    const vector<vector<float>> testImages = loadMNISTImages(testImageFile);
    testLabelFile.close();
    testImageFile.close();
    
    // === TEST GPU MODEL ===
    cout << "\n=== TESTING GPU MODEL ===" << endl;
    float accuracy_gpu = testTraining(testImages, testLabels, weights_gpu);
    
    cout << "\n=== FINAL RESULTS ===" << endl;
    cout << "GPU Accuracy: " << fixed << setprecision(2) << accuracy_gpu << "%" << endl;
    cout << "Training Time: " << duration_gpu.count() / 1000.0 << " seconds" << endl;
    cout << endl;
    
    // Save trained weights
    saveWeights("gpu_weights_cublas", weights_gpu);
    
    return 0;
}
