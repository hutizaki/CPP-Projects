#include "../bryan-library/matrixMath.h"
#include <cmath>
#include <vector>
#include <random>

using namespace std;
using namespace mm;

namespace ml {

    // z stands for layer output
    float sigmoid(const float& z) { 
        return 1.0f / (1.0f + exp(-z));
    }

    vector<float> sigmoid(const vector<float>& z) {
        vector<float> activationOutput;
        for (int i = 0; i < z.size(); i++) {
            activationOutput.push_back(sigmoid(z[i]));
        }
        return activationOutput;
    }

    float relu(const float& z) {
        return max(0.0f, z);
    }

    vector<float> relu(const vector<float>& z) {
        vector<float> activationOutput;
        for (int i = 0; i < z.size(); i++) {
            activationOutput.push_back(relu(z[i]));
        }
        return activationOutput;
    }

    vector<float> softmax(const vector<float> &z) {
      // Find max number
      float maxVal = *max_element(z.begin(), z.end());

      // Compute exp(z - max)
      vector<float> expVals(z.size());
      float sum = 0.0f;
      for (int i = 0; i < z.size(); i++) {
        expVals[i] = exp(z[i] - maxVal);
        sum += expVals[i];
      }

      // Normalize
      for (int i = 0; i < z.size(); i++) {
        expVals[i] /= sum;
      }

      return expVals;
    }

    float categoricalCrossEntropy(const vector<float> &predictions,
                                  int trueLabel) {
      // Loss = -log(prediction for true class)
      return -log(predictions[trueLabel] + 1e-7); // Add epsilon to avoid log(0)
    }

    float binaryCrossEntropy(float& y_hat, float& y) {

        // Tiny number to ensure answer isn't 1.0 or 0.0
        float epsilon = 1e-7; // 0.0000001

        // Modify y_hat with epsilon to avoid NaN
        y_hat = max(epsilon, min(1.0f - epsilon, y_hat));

        // L(a) = -[y * ln(a) + (1 - y) * ln(1 - a)]
        return -(y * log(y_hat) + (1 - y) * log(1 - y_hat));
    }

    vector<float> processLayer(vector<vector<float>>& weights, vector<float>& bias, const vector<float>& input) {
        vector<float> output = matrixVectorMultiplication(weights, input);
        return vectorAddition(output, bias);
    }

    float randomWeight() {
        static random_device rd;
        static mt19937 gen(rd());
        // Use smaller range for better initialization
        static uniform_real_distribution<float> dis(-0.3f, 0.3f);
        return dis(gen);
    }

    vector<float> generateRandVector(int numItems) {
        vector<float> row;
        for (int j = 0; j < numItems; j++) {
            row.push_back(randomWeight());
        }
        return row;
    }

    vector<vector<float>> generateRandMatrix(int numRows, int numCols) {
        vector<vector<float>> weights;
        for (int i = 0; i < numRows; i++) {
            weights.push_back(generateRandVector(numCols));
        }
        return weights;
    }

    vector<float> relu_derivative(const vector<float>& z) {
        int size = z.size();
        vector<float> relu_derivative(size);
        for (int i = 0; i < size; i++) {
            relu_derivative[i] = (z[i] > 0) ? 1.0f : 0.0f;
        }
        return relu_derivative;
    }
}
