#include "cudaMachineLearning.h"
#include "cudaMatrixMath.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <iostream>
#include <cmath>

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

// Check for kernel launch errors
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA kernel error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA kernel error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

namespace cudaML {
    // Batched matrix-vector multiplication: process batch_size vectors at once
    // Input: batch_input [batch_size x input_dim] in row-major order
    // Matrix: weights [output_dim x input_dim] in row-major order
    // Output: batch_output [batch_size x output_dim] in row-major order
    __global__ void batchedMatVecMulKernel(float* matrix, float* batch_input, float* batch_output, 
                                           int batch_size, int output_dim, int input_dim) {
        int batch_idx = blockIdx.y;
        int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (batch_idx < batch_size && output_idx < output_dim) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; i++) {
                sum += matrix[output_idx * input_dim + i] * batch_input[batch_idx * input_dim + i];
            }
            batch_output[batch_idx * output_dim + output_idx] = sum;
        }
    }

    // Batched ReLU activation
    __global__ void batchedReLUKernel(float* input, float* output, int total_elements) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < total_elements) {
            output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
        }
    }

    // Batched bias addition
    __global__ void batchedBiasAddKernel(float* input, float* bias, float* output, 
                                         int batch_size, int dim) {
        int batch_idx = blockIdx.x;
        int idx = threadIdx.x;
        
        if (batch_idx < batch_size && idx < dim) {
            output[batch_idx * dim + idx] = input[batch_idx * dim + idx] + bias[idx];
        }
    }

    std::vector<float> processLayer(const std::vector<std::vector<float>>& weights, 
                                    const std::vector<float>& bias, 
                                    const std::vector<float>& input)
    {
        std::vector<float> output = cudaMM::matrixVectorMultiplication(weights, input);
        return cudaMM::vectorAddition(output, bias);
    }

    // Batched forward pass: process entire batch on GPU
    void batchedForwardPass(cudaMM::GPUWeights& weights, float* d_batch_input, int batch_size,
                            float* d_hiddenZ, float* d_hiddenA, float* d_outputZ) {
        int threadsPerBlock = 256;
        
        // Layer 1: input -> hidden
        // Use 2D grid: (output_dim blocks in x, batch_size blocks in y)
        int numBlocks_x = (weights.w1_rows + threadsPerBlock - 1) / threadsPerBlock;
        dim3 grid1(numBlocks_x, batch_size);
        batchedMatVecMulKernel<<<grid1, threadsPerBlock>>>(
            weights.d_w1, d_batch_input, d_hiddenZ, batch_size, weights.w1_rows, weights.w1_cols);
        CUDA_CHECK_KERNEL();
        
        // Add bias and apply ReLU
        batchedBiasAddKernel<<<batch_size, weights.b1_size>>>(
            d_hiddenZ, weights.d_b1, d_hiddenZ, batch_size, weights.b1_size);
        CUDA_CHECK_KERNEL();
        
        int total_hidden = batch_size * weights.b1_size;
        int numBlocks_relu = (total_hidden + threadsPerBlock - 1) / threadsPerBlock;
        batchedReLUKernel<<<numBlocks_relu, threadsPerBlock>>>(
            d_hiddenZ, d_hiddenA, total_hidden);
        CUDA_CHECK_KERNEL();

        // Layer 2: hidden -> output
        int numBlocks_x2 = (weights.w2_rows + threadsPerBlock - 1) / threadsPerBlock;
        dim3 grid2(numBlocks_x2, batch_size);
        batchedMatVecMulKernel<<<grid2, threadsPerBlock>>>(
            weights.d_w2, d_hiddenA, d_outputZ, batch_size, weights.w2_rows, weights.w2_cols);
        CUDA_CHECK_KERNEL();
        
        batchedBiasAddKernel<<<batch_size, weights.b2_size>>>(
            d_outputZ, weights.d_b2, d_outputZ, batch_size, weights.b2_size);
        CUDA_CHECK_KERNEL();
    }

    // Batched softmax activation
    __global__ void batchedSoftmaxKernel(float* input, float* output, int batch_size, int n) {
        int batch_idx = blockIdx.x;
        
        if (batch_idx < batch_size) {
            // Find max for numerical stability (within this batch sample)
            float maxVal = input[batch_idx * n];
            for (int i = 1; i < n; i++) {
                if (input[batch_idx * n + i] > maxVal) {
                    maxVal = input[batch_idx * n + i];
                }
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                output[batch_idx * n + i] = expf(input[batch_idx * n + i] - maxVal);
                sum += output[batch_idx * n + i];
            }
            
            // Normalize
            for (int i = 0; i < n; i++) {
                output[batch_idx * n + i] /= sum;
            }
        }
    }

    // Compute output error: y_hat - one_hot(trueLabel)
    __global__ void computeOutputErrorKernel(float* y_hat, int* trueLabels, float* outputError,
                                            int batch_size, int n) {
        int batch_idx = blockIdx.x;
        int idx = threadIdx.x;
        
        if (batch_idx < batch_size && idx < n) {
            int trueLabel = trueLabels[batch_idx];
            outputError[batch_idx * n + idx] = y_hat[batch_idx * n + idx] - (idx == trueLabel ? 1.0f : 0.0f);
        }
    }

    // Compute gradients for W2: dW2 = outputError * hiddenA^T (outer product)
    __global__ void computeGradientW2Kernel(float* outputError, float* hiddenA, float* dW2,
                                            int batch_size, int output_dim, int hidden_dim) {
        int batch_idx = blockIdx.z;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (batch_idx < batch_size && row < output_dim && col < hidden_dim) {
            // Accumulate gradient: dW2[row][col] += outputError[batch][row] * hiddenA[batch][col]
            atomicAdd(&dW2[row * hidden_dim + col], 
                     outputError[batch_idx * output_dim + row] * hiddenA[batch_idx * hidden_dim + col]);
        }
    }

    // Compute gradients for b2: db2 = outputError (sum over batch)
    __global__ void computeGradientB2Kernel(float* outputError, float* db2,
                                           int batch_size, int dim) {
        int batch_idx = blockIdx.x;
        int idx = threadIdx.x;
        
        if (batch_idx < batch_size && idx < dim) {
            atomicAdd(&db2[idx], outputError[batch_idx * dim + idx]);
        }
    }

    // Backpropagate to hidden layer: hiddenError = W2^T * outputError
    __global__ void backpropToHiddenKernel(float* W2, float* outputError, float* hiddenError,
                                           int batch_size, int hidden_dim, int output_dim) {
        int batch_idx = blockIdx.y;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (batch_idx < batch_size && idx < hidden_dim) {
            float sum = 0.0f;
            // W2^T: transpose means W2[j][idx] = W2[j * hidden_dim + idx]
            for (int j = 0; j < output_dim; j++) {
                sum += outputError[batch_idx * output_dim + j] * W2[j * hidden_dim + idx];
            }
            hiddenError[batch_idx * hidden_dim + idx] = sum;
        }
    }

    // Apply ReLU derivative: zero out gradients where hiddenZ <= 0
    __global__ void reluBackwardKernel(float* hiddenZ, float* hiddenError, int total_elements) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (idx < total_elements) {
            if (hiddenZ[idx] <= 0.0f) {
                hiddenError[idx] = 0.0f;
            }
        }
    }

    // Compute gradients for W1: dW1 = hiddenError * input^T (outer product)
    __global__ void computeGradientW1Kernel(float* hiddenError, float* input, float* dW1,
                                           int batch_size, int hidden_dim, int input_dim) {
        int batch_idx = blockIdx.z;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (batch_idx < batch_size && row < hidden_dim && col < input_dim) {
            // Accumulate gradient: dW1[row][col] += hiddenError[batch][row] * input[batch][col]
            atomicAdd(&dW1[row * input_dim + col],
                     hiddenError[batch_idx * hidden_dim + row] * input[batch_idx * input_dim + col]);
        }
    }

    // Compute gradients for b1: db1 = hiddenError (sum over batch)
    __global__ void computeGradientB1Kernel(float* hiddenError, float* db1,
                                          int batch_size, int dim) {
        int batch_idx = blockIdx.x;
        int idx = threadIdx.x;
        
        if (batch_idx < batch_size && idx < dim) {
            atomicAdd(&db1[idx], hiddenError[batch_idx * dim + idx]);
        }
    }

    // Update weights: W -= learning_rate * (gradient / batch_size)
    __global__ void updateWeightsKernel(float* weights, float* gradients, float learning_rate,
                                       int batch_size, int total_elements) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (idx < total_elements) {
            weights[idx] -= learning_rate * (gradients[idx] / batch_size);
        }
    }

    // Update bias: bias -= learning_rate * (gradient / batch_size)
    __global__ void updateBiasKernel(float* bias, float* gradients, float learning_rate,
                                    int batch_size, int dim) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (idx < dim) {
            bias[idx] -= learning_rate * (gradients[idx] / batch_size);
        }
    }

    // Batched backward pass: compute gradients and update weights on GPU
    void batchedBackwardPass(cudaMM::GPUWeights& weights, 
                            float* d_batch_input, int* d_trueLabels,
                            float* d_hiddenZ, float* d_hiddenA, float* d_outputZ, float* d_y_hat,
                            float* d_dW1, float* d_db1, float* d_dW2, float* d_db2,
                            int batch_size, float learning_rate) {
        int threadsPerBlock = 256;
        
        // Step 1: Apply softmax to outputZ -> y_hat
        batchedSoftmaxKernel<<<batch_size, 1>>>(d_outputZ, d_y_hat, batch_size, weights.b2_size);
        CUDA_CHECK_KERNEL();
        
        // Allocate temporary device memory for errors
        float *d_outputError, *d_hiddenError;
        CUDA_CHECK(cudaMalloc(&d_outputError, batch_size * weights.b2_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hiddenError, batch_size * weights.b1_size * sizeof(float)));
        
        // Step 2: Compute output error: y_hat - one_hot(trueLabel)
        computeOutputErrorKernel<<<batch_size, weights.b2_size>>>(
            d_y_hat, d_trueLabels, d_outputError, batch_size, weights.b2_size);
        CUDA_CHECK_KERNEL();
        
        // Zero out gradient accumulators
        CUDA_CHECK(cudaMemset(d_dW1, 0, weights.w1_rows * weights.w1_cols * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_db1, 0, weights.b1_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dW2, 0, weights.w2_rows * weights.w2_cols * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_db2, 0, weights.b2_size * sizeof(float)));
        
        // Step 3: Compute gradients for W2 and b2
        dim3 threadsW2(16, 16);
        dim3 blocksW2((weights.w2_cols + 15) / 16, (weights.w2_rows + 15) / 16, batch_size);
        computeGradientW2Kernel<<<blocksW2, threadsW2>>>(
            d_outputError, d_hiddenA, d_dW2, batch_size, weights.w2_rows, weights.w2_cols);
        CUDA_CHECK_KERNEL();
        
        computeGradientB2Kernel<<<batch_size, weights.b2_size>>>(
            d_outputError, d_db2, batch_size, weights.b2_size);
        CUDA_CHECK_KERNEL();
        
        // Step 4: Backpropagate to hidden layer
        int numBlocks_hidden = (weights.b1_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 grid_hidden(numBlocks_hidden, batch_size);
        backpropToHiddenKernel<<<grid_hidden, threadsPerBlock>>>(
            weights.d_w2, d_outputError, d_hiddenError, batch_size, weights.b1_size, weights.w2_rows);
        CUDA_CHECK_KERNEL();
        
        // Step 5: Apply ReLU derivative
        int total_hidden = batch_size * weights.b1_size;
        int numBlocks_relu = (total_hidden + threadsPerBlock - 1) / threadsPerBlock;
        reluBackwardKernel<<<numBlocks_relu, threadsPerBlock>>>(
            d_hiddenZ, d_hiddenError, total_hidden);
        CUDA_CHECK_KERNEL();
        
        // Step 6: Compute gradients for W1 and b1
        dim3 threadsW1(16, 16);
        dim3 blocksW1((weights.w1_cols + 15) / 16, (weights.w1_rows + 15) / 16, batch_size);
        computeGradientW1Kernel<<<blocksW1, threadsW1>>>(
            d_hiddenError, d_batch_input, d_dW1, batch_size, weights.w1_rows, weights.w1_cols);
        CUDA_CHECK_KERNEL();
        
        computeGradientB1Kernel<<<batch_size, weights.b1_size>>>(
            d_hiddenError, d_db1, batch_size, weights.b1_size);
        CUDA_CHECK_KERNEL();
        
        // Step 7: Update weights on GPU
        int total_w1 = weights.w1_rows * weights.w1_cols;
        int numBlocks_w1 = (total_w1 + threadsPerBlock - 1) / threadsPerBlock;
        updateWeightsKernel<<<numBlocks_w1, threadsPerBlock>>>(
            weights.d_w1, d_dW1, learning_rate, batch_size, total_w1);
        CUDA_CHECK_KERNEL();
        
        int total_w2 = weights.w2_rows * weights.w2_cols;
        int numBlocks_w2 = (total_w2 + threadsPerBlock - 1) / threadsPerBlock;
        updateWeightsKernel<<<numBlocks_w2, threadsPerBlock>>>(
            weights.d_w2, d_dW2, learning_rate, batch_size, total_w2);
        CUDA_CHECK_KERNEL();
        
        // Step 8: Update biases on GPU
        int numBlocks_b1 = (weights.b1_size + threadsPerBlock - 1) / threadsPerBlock;
        updateBiasKernel<<<numBlocks_b1, threadsPerBlock>>>(
            weights.d_b1, d_db1, learning_rate, batch_size, weights.b1_size);
        CUDA_CHECK_KERNEL();
        
        int numBlocks_b2 = (weights.b2_size + threadsPerBlock - 1) / threadsPerBlock;
        updateBiasKernel<<<numBlocks_b2, threadsPerBlock>>>(
            weights.d_b2, d_db2, learning_rate, batch_size, weights.b2_size);
        CUDA_CHECK_KERNEL();
        
        // Free temporary memory
        CUDA_CHECK(cudaFree(d_outputError));
        CUDA_CHECK(cudaFree(d_hiddenError));
    }
}