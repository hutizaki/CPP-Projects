#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

namespace cudaMM{
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

    // Flattens a 2D matrix to a 1D array
    vector<float> rowMajorMatrix(const vector<vector<float>> &matrix) {
        int rowSize = matrix[0].size();
        int colSize = matrix.size();

        vector<float> final;
        for (int i = 0; i < colSize; i++) {
            for (int j = 0; j < rowSize; j++) {
            final.push_back(matrix[i][j]);
            }
        }

        return final;
    }

    vector<float> matrixVectorMultiplication(const vector<vector<float>>&A, const vector<float>& Vector) {
        int Ak = A[0].size();
        int m = A.size();
        int k = Vector.size();


        if (Ak != k) {
          throw runtime_error(
              "Matrix and Vector must follow [m x k] * [k] format");
        }

        vector<float> flattenedA = rowMajorMatrix(A);
        vector<float> result(m);

        int sizeA = sizeof(flattenedA);
        int sizeV = sizeof(Vector);
        int sizeR = sizeof(result);

        float *d_A, *d_V, *d_R;

        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_V, sizeV);
        cudaMalloc(&d_R, sizeR);

        cudaMemcpy(d_A, flattenedA.data(), sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, Vector.data(), sizeV, cudaMemcpyHostToDevice);
        cudaMemcpy(d_R, result.data(), sizeR, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;

        matVecMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_V, d_R, m, Ak);
        cudaDeviceSynchronize();

        cudaMemcpy(result.data(), d_R, sizeR, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_V);
        cudaFree(d_R);

        return result;
    }
}