#ifndef MACHINELEARNING_H
#define MACHINELEARNING_H

#include <vector>
#include <iostream>

using namespace std;

namespace ml {
    // Activation functions
    float sigmoid(const float& z);
    vector<float> sigmoid(const vector<float>& z);
    float relu(const float& z);
    vector<float> relu(const vector<float>& z);
    vector<float> softmax(const vector<float>& z);

    // Loss functions
    float binaryCrossEntropy(float& predicted, float& trueLabel);
    float categoricalCrossEntropy(const vector<float>& predictions, int trueLabel);

    vector<float> processLayer(vector<vector<float>>& weights, vector<float>& bias, const vector<float>& input);
    float randomWeight();
    vector<vector<float>> generateRandMatrix(int numRows, int numCols);
    vector<float> generateRandVector(int numItems);
    vector<float> relu_derivative(const vector<float>& z);
}

#endif

