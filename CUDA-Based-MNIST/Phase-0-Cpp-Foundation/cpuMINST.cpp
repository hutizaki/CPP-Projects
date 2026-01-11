#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

float sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
}

float linearCombination(float x1, float x2, float w1, float w2, float b) {
    return w1 * x1 + w2 * x2 + b;
}

float binaryCrossEntropy(float y_hat, int y) {
  // Avoid log(0) by adding tiny epsilon
  float epsilon = 1e-7;
  y_hat = max(epsilon, min(1.0f - epsilon, y_hat));

  return -(y * log(y_hat) + (1 - y) * log(1 - y_hat));
}

struct Sample {
    float x1;
    float x2;
    int label;
};

void runTraining(vector<Sample> samples, int epochs, float learning_rate, string function_name) {
    float w1 = 8.55f;
    float w2 = 8.55f;
    float b = -12.99f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        for (const Sample& sample : samples) {

            float z = linearCombination(sample.x1, sample.x2, w1, w2, b);

            float y_hat = sigmoid(z);
            float loss = binaryCrossEntropy(y_hat, sample.label);
            total_loss += loss;

            float error = y_hat - sample.label;
            float dw1 = error * sample.x1;
            float dw2 = error * sample.x2;
            float db = error;

            w1 -= learning_rate * dw1;
            w2 -= learning_rate * dw2;
            b -= learning_rate * db;
        }

        float avg_loss = total_loss / samples.size();

        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << ", Loss: " << avg_loss << endl;
        }

        // if (avg_loss < 0.01) {
        //     cout << "Training completed in " << epoch << " epochs" << endl;
        //     break;
        // }
    }

    ofstream outputFile(function_name + "_weights.txt");
    outputFile << "Final weights:\nw1 = " << w1 << "\nw2 = " << w2 << "\nb = " << b << endl;
    outputFile << "Final predictions:" << endl;
    for (const Sample &s : samples) {
        float z = w1 * s.x1 + w2 * s.x2 + b;
        float y_hat = sigmoid(z);
        int predicted = (y_hat >= 0.5) ? 1 : 0;
        outputFile << "Input: (" << s.x1 << ", " << s.x2 << ") Predicted: " << y_hat << " (" << predicted << ") Actual: " << s.label << (predicted == s.label ? " ✓" : " ✗") << endl;
    }
    return;
}

int main() {
    vector<Sample> samples;
    string line;
    string function_name;
    cin >> function_name;
    
    // Read from standard input (cin)
    while (getline(cin, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        stringstream currStream(line);
        float x1, x2;
        int label;
        currStream >> x1 >> x2 >> label;
        Sample s = {x1, x2, label};
        samples.push_back(s);
    }

    runTraining(samples, 1000000000, 0.5, function_name);

    return 0;
}