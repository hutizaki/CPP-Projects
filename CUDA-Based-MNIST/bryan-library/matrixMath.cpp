#include "matrixMath.h"
#include <stdexcept>
#include <vector>
#include <iostream>

using namespace std;

namespace mm {

vector<float> colFromMatrix(const vector<vector<float>>& matrix, int colNum) {

  vector<float> column;
  for (int row = 0; row < matrix.size(); row++) {
    column.push_back(matrix[row][colNum]);
  }

  return column;
}

vector<float> rowFromMatrix(const vector<vector<float>>& matrix, int rowNum) {

    vector<float> row;
    for (int col = 0; col < matrix.size(); col++) {
        row.push_back(matrix[rowNum][col]);
    }
    return row;
}

/*
m x k         k x n
    [1 2 3]   [1]   [1*1 + 2*2 + 3*3]   [14]
    [4 5 6] × [2] = [4*1 + 5*2 + 6*3] = [32]
              [3]
*/


vector<vector<float>> matrixMultiplication(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    
    int Ak = A[0].size();
    int Bk = B.size();
    int m = A.size();
    int n = B[0].size();
    
    if (Ak != Bk) {
        throw runtime_error("Matrices must follow [m x k] * [k x n] format");
    }

    vector<vector<float>> answer;
    
    for (int r = 0; r < m; r++) {
        const vector<float> *prow = &A[r];
        vector<float> rowResult;
        for (int i = 0; i < n; i++) {
            vector<float> column = colFromMatrix(B, i);
            float sum = 0.0f;
            for (int x = 0; x < Ak; x++) {
                sum += (*prow)[x] * column[x];
            }
            rowResult.push_back(sum);
        }
        answer.push_back(rowResult);
    }

    return answer;
}

vector<float> matrixVectorMultiplication(const vector<vector<float>> &A,
                                           const vector<float> &Vector) {

  int Ak = A[0].size();
  int m = A.size();
  int k = Vector.size();

  if (Ak != k) {
    throw runtime_error("Matrix and Vector must follow [m x k] * [k] format");
  }

  vector<float> answer;

  for (int r = 0; r < m; r++) {
    float sum = 0.0f;
    for (int x = 0; x < Ak; x++) {
      sum += A[r][x] * Vector[x];
    }
    answer.push_back(sum);
  }

  return answer;
}

vector<vector<float>> vectorMultiplication(const vector<float> &V1, const vector<float> &V2) {
  int sizeV1 = V1.size();
  int sizeV2 = V2.size();

  vector<vector<float>> matrix(sizeV1, vector<float>(sizeV2));

  for (int i = 0; i < sizeV1; i++) {
    for (int j = 0; j < sizeV2; j++) {
      matrix[i][j] = V1[i] * V2[j];
    }
  }

  return matrix;
}

vector<float> vectorAddition(const vector<float>& z, const vector<float>& bias) {
    int size = z.size();

    if (size != bias.size()) {
        throw runtime_error("Cannot complete addition, the two vectors must be equal size.");
    }

    vector<float> answer;

    for (int i = 0; i < size; i++) {
        answer.push_back(z[i] + bias[i]);
    }

    return answer;
}

void printMatrix(vector<vector<float>> matrix) {
  for (auto row : matrix) {
    cout << "[ ";
    for (size_t j = 0; j < row.size(); j++) {
      cout << row[j];
      if (j < row.size() - 1) {
        cout << "  ";
      }
    }
    cout << " ]" << endl;
  }
}

void printVector(vector<float> &vec) {
  cout << "[ ";
  for (size_t i = 0; i < vec.size(); i++) {
    cout << vec[i];
    if (i < vec.size() - 1) {
      cout << "  ";
    }
  }
  cout << " ]" << endl;
}

void printVector(vector<int> &vec) {
  cout << "[ ";
  for (size_t i = 0; i < vec.size(); i++) {
    cout << vec[i];
    if (i < vec.size() - 1) {
      cout << "  ";
    }
  }
  cout << " ]" << endl;
}

vector<vector<float>> transpose(const vector<vector<float>>& matrix) {
    if (matrix.empty()) {
        return {};
    }
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    // Create transposed matrix: cols × rows
    vector<vector<float>> transposed(cols, vector<float>(rows));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j][i] = matrix[i][j];  // Swap indices
        }
    }
    
    return transposed;
}

// Test main function - commented out to avoid linker conflicts
// Uncomment this if you want to test matrixMath.cpp independently
/*
int main() {
    const vector<vector<float>> matrix1 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const vector<vector<float>> matrix3 = {
        {1.0, 2.0}, 
        {3.0, 4.0}
    };

    const vector<float> matrix2 = {
        1.0,
        2.0
    };

    vector<vector<float>> answer = matrixMultiplication(matrix1, matrix3);
    printMatrix(answer);

    cout << endl;

    vector<float> answer2 = matrixVectorMultiplication(matrix1, matrix2);
    printVector(answer2);
    return 0;
}
*/

} // namespace mm