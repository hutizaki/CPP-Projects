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

    vector<float> matrixVectorMultiplication(const vector<vector<float>> )

    // Flattens a 2D matrix to a 1D array
    vector<float> rowMajorMatrix(const vector<vector<float>>& matrix) {
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
}