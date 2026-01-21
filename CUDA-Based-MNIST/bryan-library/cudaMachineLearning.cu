#include "cudaMachineLearning.h"
#include "cudaMatrixMath.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <iostream>

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
}