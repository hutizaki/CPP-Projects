#ifndef MATRIXMATH_H
#define MATRIXMATH_H

#include <vector>
#include <iostream>

using namespace std;

namespace mm {
    vector<float> colFromMatrix(const vector<vector<float>>& matrix, int colNum);
    vector<float> rowFromMatrix(const vector<vector<float>>& matrix, int rowNum);
    vector<vector<float>> matrixMultiplication(const vector<vector<float>>& A, const vector<vector<float>>& B);
    vector<float> matrixVectorMultiplication(const vector<vector<float>>& A, const vector<float>& Vector);
    vector<vector<float>> vectorMultiplication(const vector<float>& V1, const vector<float>& V2);
    vector<float> vectorAddition(const vector<float>& A, const vector<float>& B);
    vector<vector<float>> transpose(const vector<vector<float>>& matrix);
    void printMatrix(vector<vector<float>> matrix);
    void printVector(vector<float> &vec);
    void printVector(vector<int> &vec);
}

#endif

