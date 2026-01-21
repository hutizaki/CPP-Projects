#ifndef CUDAMATRIXMATH_H
#define CUDAMATRIXMATH_H

#include <vector>

namespace cudaMM {
    /// @brief Flattens a 2D matrix to a 1D array (row-major order)
    /// @param matrix The 2D matrix to flatten
    /// @return A 1D vector containing the matrix elements in row-major order
    std::vector<float> rowMajorMatrix(const std::vector<std::vector<float>>& matrix);
    
    /// @brief CUDA-accelerated matrix-vector multiplication
    /// @param A The matrix [m x k]
    /// @param Vector The vector [k]
    /// @return The result vector [m]
    /// @throws std::runtime_error if dimensions don't match (Ak != k)
    /// @details Performs: [m x k] matrix * [k] vector = [m] result
    std::vector<float> matrixVectorMultiplication(
        const std::vector<std::vector<float>>& A, 
        const std::vector<float>& Vector);

    /// @brief CUDA-accelerated vector addition
    /// @param A First vector [n]
    /// @param B Second vector [n]
    /// @return The result vector [n] containing element-wise sum A + B
    /// @throws std::runtime_error if dimensions don't match (A.size() != B.size())
    /// @details Performs element-wise addition: result[i] = A[i] + B[i]
    std::vector<float> vectorAddition(
        const std::vector<float>& A, 
        const std::vector<float>& B);

    /// @brief GPU Weight Manager - keeps weights on GPU to minimize transfers
    class GPUWeights {
    public:
        float* d_w1;
        float* d_w2;
        float* d_b1;
        float* d_b2;
        int w1_rows, w1_cols;
        int w2_rows, w2_cols;
        int b1_size, b2_size;

        GPUWeights(const std::vector<std::vector<float>>& w1_host,
                   const std::vector<std::vector<float>>& w2_host,
                   const std::vector<float>& b1_host,
                   const std::vector<float>& b2_host);
        ~GPUWeights();
        void copyToHost(std::vector<std::vector<float>>& w1_host,
                       std::vector<std::vector<float>>& w2_host,
                       std::vector<float>& b1_host,
                       std::vector<float>& b2_host);
    };

}

#endif // CUDAMATRIXMATH_H