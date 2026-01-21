#include "cudaMatrixMath.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <algorithm>
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

namespace cudaMM {
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

    __global__ void vecAddKernel(float* vectorA, float* vectorB, float* result, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < n) {
            result[idx] = vectorA[idx] + vectorB[idx];
        }
    }

    // Flattens a 2D matrix to a 1D array
    std::vector<float> rowMajorMatrix(const std::vector<std::vector<float>>& matrix) {
        int rowSize = matrix[0].size();
        int colSize = matrix.size();

        std::vector<float> final;
        for (int i = 0; i < colSize; i++) {
            for (int j = 0; j < rowSize; j++) {
                final.push_back(matrix[i][j]);
            }
        }

        return final;
    }

    std::vector<float> matrixVectorMultiplication(
        const std::vector<std::vector<float>>& A, 
        const std::vector<float>& Vector) {
        int Ak = A[0].size();
        int m = A.size();
        int k = Vector.size();

        if (Ak != k) {
            throw std::runtime_error(
                "Matrix and Vector must follow [m x k] * [k] format");
        }

        std::vector<float> flattenedA = rowMajorMatrix(A);
        std::vector<float> result(m);

        int aSize = flattenedA.size() * sizeof(float);
        int vSize = Vector.size() * sizeof(float);
        int rSize = result.size() * sizeof(float);

        float *d_A, *d_V, *d_R;

        CUDA_CHECK(cudaMalloc(&d_A, aSize));
        CUDA_CHECK(cudaMalloc(&d_V, vSize));
        CUDA_CHECK(cudaMalloc(&d_R, rSize));

        CUDA_CHECK(cudaMemcpy(d_A, flattenedA.data(), aSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, Vector.data(), vSize, cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;

        matVecMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_V, d_R, m, Ak);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(result.data(), d_R, rSize, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_R));

        return result;
    }

    std::vector<float> vectorAddition(
        const std::vector<float>& A, 
        const std::vector<float>& B) {

        if (A.size() != B.size()) {
            throw std::runtime_error("Cannot complete addition, the two vectors must be equal size.");
        }

        int arrayLength = A.size();
        int size = arrayLength * sizeof(float);

        std::vector<float> result(arrayLength);

        float *d_A, *d_B, *d_R;

        CUDA_CHECK(cudaMalloc(&d_A, size));
        CUDA_CHECK(cudaMalloc(&d_B, size));
        CUDA_CHECK(cudaMalloc(&d_R, size));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice));

        int threadsPerBlock = std::min(256, arrayLength);
        int numBlocks = (arrayLength + threadsPerBlock - 1) / threadsPerBlock;

        vecAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_R, arrayLength);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(result.data(), d_R, size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_R));

        return result;
    }

    // GPU Weight Manager - keeps weights on GPU to avoid repeated transfers
    GPUWeights::GPUWeights(const std::vector<std::vector<float>>& w1_host,
                           const std::vector<std::vector<float>>& w2_host,
                           const std::vector<float>& b1_host,
                           const std::vector<float>& b2_host) {
        w1_rows = w1_host.size();
            w1_cols = w1_host[0].size();
            w2_rows = w2_host.size();
            w2_cols = w2_host[0].size();
            b1_size = b1_host.size();
            b2_size = b2_host.size();

            // Flatten and allocate weights on GPU
            std::vector<float> w1_flat = rowMajorMatrix(w1_host);
            std::vector<float> w2_flat = rowMajorMatrix(w2_host);

            CUDA_CHECK(cudaMalloc(&d_w1, w1_rows * w1_cols * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_w2, w2_rows * w2_cols * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_b1, b1_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_b2, b2_size * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_w1, w1_flat.data(), w1_rows * w1_cols * sizeof(float), 
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_w2, w2_flat.data(), w2_rows * w2_cols * sizeof(float), 
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b1, b1_host.data(), b1_size * sizeof(float), 
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b2, b2_host.data(), b2_size * sizeof(float), 
                                  cudaMemcpyHostToDevice));
        }

    GPUWeights::~GPUWeights() {
        if (d_w1) cudaFree(d_w1);
        if (d_w2) cudaFree(d_w2);
        if (d_b1) cudaFree(d_b1);
        if (d_b2) cudaFree(d_b2);
    }

    // Copy weights back to host
    void GPUWeights::copyToHost(std::vector<std::vector<float>>& w1_host,
                               std::vector<std::vector<float>>& w2_host,
                               std::vector<float>& b1_host,
                               std::vector<float>& b2_host) {
            std::vector<float> w1_flat(w1_rows * w1_cols);
            std::vector<float> w2_flat(w2_rows * w2_cols);

            CUDA_CHECK(cudaMemcpy(w1_flat.data(), d_w1, w1_rows * w1_cols * sizeof(float), 
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(w2_flat.data(), d_w2, w2_rows * w2_cols * sizeof(float), 
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b1_host.data(), d_b1, b1_size * sizeof(float), 
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b2_host.data(), d_b2, b2_size * sizeof(float), 
                                  cudaMemcpyDeviceToHost));

            // Reshape w1
            w1_host.resize(w1_rows);
            for (int i = 0; i < w1_rows; i++) {
                w1_host[i].resize(w1_cols);
                for (int j = 0; j < w1_cols; j++) {
                    w1_host[i][j] = w1_flat[i * w1_cols + j];
                }
            }

            // Reshape w2
            w2_host.resize(w2_rows);
            for (int i = 0; i < w2_rows; i++) {
                w2_host[i].resize(w2_cols);
                for (int j = 0; j < w2_cols; j++) {
                    w2_host[i][j] = w2_flat[i * w2_cols + j];
                }
            }
        }

}