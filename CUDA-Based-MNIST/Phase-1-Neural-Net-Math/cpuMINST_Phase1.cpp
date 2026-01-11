#include "../bryan-library/matrixMath.h"
#include "../bryan-library/machineLearning.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <random>

using namespace ml;
using namespace mm;
using namespace std;

vector<float> ReLU(vector<float>& zWithBias) {
    vector<float> answer;
    for (int i = 0; i < zWithBias.size(); i++) {
        answer.push_back(max(0.0f, zWithBias[i]));
    }
    return answer;
}

void runTraining(vector<vector<float>> samples, int epochs, float learning_rate, string function_name) {
    // Initialize weights and biases with small random values
    // This is critical - zero initialization prevents learning!
    vector<vector<float>> w1 = {
        {2, 2},
        {randomWeight(), randomWeight()}
    };

    vector<vector<float>> w2 = {
        {randomWeight(), randomWeight()}
    };

    vector<float> b1 = {randomWeight(), randomWeight()};  // Hidden layer biases
    vector<float> b2 = {randomWeight()};                 // Output layer bias

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        for (const vector<float> & sample : samples) {
            // Extract inputs and label
            const vector<float> input = {sample[0], sample[1]};
            float y = sample[2];

            // ========== FORWARD PASS ==========
            // Hidden layer
            const vector<float> z_hidden = matrixVectorMultiplication(w1, input);
            vector<float> z_hidden_bias = vectorAddition(z_hidden, b1);
            vector<float> h = ReLU(z_hidden_bias);

            // Output layer
            vector<float> z_output = processLayer(w2, b2, h);
            float y_hat = sigmoid(z_output[0]);
            
            // Compute loss
            float loss = binaryCrossEntropy(y_hat, y);
            total_loss += loss;

            // ========== BACKWARD PASS ==========
            // Error signal (derivative of loss w.r.t. z_output)
            // For binary cross-entropy + sigmoid: dL/dz = y_hat - y_true
            float error = y_hat - y;

            // Update W2 (output layer weights)
            for (int j = 0; j < 2; j++) {
                float dW2 = error * h[j];
                // Clip gradient to prevent exploding gradients
                dW2 = max(-5.0f, min(5.0f, dW2));
                w2[0][j] -= learning_rate * dW2;
            }

            // Update b2 (output layer bias)
            float db2 = error;
            db2 = max(-5.0f, min(5.0f, db2));
            b2[0] -= learning_rate * db2;

            // Propagate error back to hidden layer
            // dL/dh = dL/dz_output * dz_output/dh = error * W2
            vector<float> dL_dh(2);
            for (int i = 0; i < 2; i++) {
                dL_dh[i] = error * w2[0][i];
            }

            // Update W1 (hidden layer weights)
            for (int i = 0; i < 2; i++) {
                // ReLU derivative: 1 if z > 0, else 0
                float relu_derivative = (z_hidden_bias[i] > 0) ? 1.0f : 0.0f;
                
                for (int j = 0; j < 2; j++) {
                    // dL/dW1 = dL/dh * dh/dz * dz/dW1 = dL_dh * relu_derivative * input
                    float dW1 = dL_dh[i] * relu_derivative * input[j];
                    // Clip gradient
                    dW1 = max(-5.0f, min(5.0f, dW1));
                    w1[i][j] -= learning_rate * dW1;
                }
                
                // Update b1 (hidden layer bias)
                // dL/db1 = dL/dh * dh/dz * dz/db1 = dL_dh * relu_derivative * 1
                float db1 = dL_dh[i] * relu_derivative;
                db1 = max(-5.0f, min(5.0f, db1));
                b1[i] -= learning_rate * db1;
            }
        }

        float avg_loss = total_loss / samples.size();

        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << ", Loss: " << avg_loss << endl;
        }

        if (avg_loss < 0.01) {
            cout << "Training completed in " << epoch << " epochs" << endl;
            break;
        }
    }

    // Save final weights and predictions
    ofstream outputFile(function_name + "_weights.txt");
    outputFile << "Final weights:\n";
    outputFile << "W1:\n";
    for (int i = 0; i < 2; i++) {
        outputFile << "  [" << w1[i][0] << ", " << w1[i][1] << "]\n";
    }
    outputFile << "b1: [" << b1[0] << ", " << b1[1] << "]\n";
    outputFile << "W2: [" << w2[0][0] << ", " << w2[0][1] << "]\n";
    outputFile << "b2: [" << b2[0] << "]\n";
    outputFile << "\nFinal predictions:\n";
    
    for (const vector<float> &s : samples) {
        vector<float> input = {s[0], s[1]};
        float y_true = s[2];
        
        // Forward pass for prediction
        vector<float> z_hidden_bias = processLayer(w1, b1, input);
        vector<float> hidden_output = ReLU(z_hidden_bias);
        vector<float> z_output = processLayer(w2, b2, hidden_output);
        float y_hat = sigmoid(z_output[0]);
        
        int predicted = (y_hat >= 0.5) ? 1 : 0;
        int actual = (int)y_true;
        outputFile << "Input: (" << s[0] << ", " << s[1] << ") Predicted: " << y_hat << " (" << predicted << ") Actual: " << actual << (predicted == actual ? " ✓" : " ✗") << "\n";
    }
    outputFile.close();
}

int main() {
    vector<vector<float>> TestData;
    string line;
    string function_name;
    cin >> function_name;
    
    // Read from standard input (cin)
    while (getline(cin, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        stringstream currStream(line);
        float x1, x2, label;
        currStream >> x1 >> x2 >> label;
        vector<float> sample = {x1, x2, label};
        TestData.push_back(sample);
    }

    runTraining(TestData, 1000000, 0.01, function_name);

    return 0;
}