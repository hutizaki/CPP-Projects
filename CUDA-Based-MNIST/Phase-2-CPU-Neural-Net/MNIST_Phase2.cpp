#include "../bryan-library/matrixMath.h"
#include "../bryan-library/machineLearning.h"
#include "../bryan-library/fileParsing.h"
#include <cstdint>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace ml;
using namespace mm;
using namespace std;
using namespace fp;

vector<float> forwardPass(vector<vector<float>>& weights, vector<float>& bias, const vector<float>& input) {
    vector<float> hiddenOutput = processLayer(weights, bias, input);
    // vector<float> hiddenActivation;  

    // if (function == "relu") {
    //     hiddenActivation = relu(hiddenOutput);
    // } else if (function == "sigmoid") {
    //     hiddenActivation = sigmoid(hiddenOutput);
    // } else {
    //     throw runtime_error("Function type not accepted");
    // }

    return hiddenOutput;
}

struct TrainedWeights {
    vector<vector<float>> w1;
    vector<vector<float>> w2;
    vector<float> b1;
    vector<float> b2;
};

TrainedWeights runTraining(const vector<vector<float>> &images, const vector<int> labels, int numNeurons, int epochs, float learning_rate, int batch_size, string function_name)  {
    int inputLength = images[0].size();
    vector<vector<float>> w1 = generateRandMatrix(numNeurons, inputLength);
    vector<vector<float>> w2 = generateRandMatrix(10, numNeurons);
    vector<float> b1 = generateRandVector(numNeurons);
    vector<float> b2 = generateRandVector(10);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        // Loop through batches
        for (int batch_start = 0; batch_start < images.size(); batch_start += batch_size) {
            // Initialize gradient accumulators
            vector<vector<float>> dW2_batch(10, vector<float>(numNeurons, 0.0f));
            vector<float> db2_batch(10, 0.0f);
            vector<vector<float>> dW1_batch(numNeurons, vector<float>(inputLength, 0.0f));
            vector<float> db1_batch(numNeurons, 0.0f);
            
            int actual_batch_size = min(batch_size, (int)images.size() - batch_start);
            
            // Process each sample in the batch
            for (int i = 0; i < actual_batch_size; i++) {
                int idx = batch_start + i;
                
                // === FORWARD PASS ===
                vector<float> hiddenZ = processLayer(w1, b1, images[idx]);
                vector<float> hiddenA = relu(hiddenZ);
                vector<float> outputZ = processLayer(w2, b2, hiddenA);
                vector<float> y_hat = softmax(outputZ);
                
                // Compute loss
                float loss = categoricalCrossEntropy(y_hat, labels[idx]);
                total_loss += loss;
                
                // === BACKWARD PASS ===
                // Compute output error
                vector<float> outputError(10);
                for (int j = 0; j < 10; j++) {
                    outputError[j] = y_hat[j] - (j == labels[idx] ? 1.0f : 0.0f);
                }
                
                // Accumulate gradients for W2 and b2
                for (int j = 0; j < 10; j++) {
                    for (int k = 0; k < numNeurons; k++) {
                        dW2_batch[j][k] += outputError[j] * hiddenA[k];
                    }
                    db2_batch[j] += outputError[j];
                }
                
                // Backpropagate to hidden layer
                vector<float> hiddenError = matrixVectorMultiplication(transpose(w2), outputError);
                
                // Accumulate gradients for W1 and b1
                for (int j = 0; j < numNeurons; j++) {
                    float relu_deriv = (hiddenZ[j] > 0) ? 1.0f : 0.0f;
                    float dL_dz1 = hiddenError[j] * relu_deriv;
                    
                    for (int k = 0; k < inputLength; k++) {
                        dW1_batch[j][k] += dL_dz1 * images[idx][k];
                    }
                    db1_batch[j] += dL_dz1;
                }
            }
            
            // === UPDATE WEIGHTS (once per batch) ===
            // Update W2 and b2
            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < numNeurons; k++) {
                    w2[j][k] -= learning_rate * (dW2_batch[j][k] / actual_batch_size);
                }
                b2[j] -= learning_rate * (db2_batch[j] / actual_batch_size);
            }
            
            // Update W1 and b1
            for (int j = 0; j < numNeurons; j++) {
                for (int k = 0; k < inputLength; k++) {
                    w1[j][k] -= learning_rate * (dW1_batch[j][k] / actual_batch_size);
                }
                b1[j] -= learning_rate * (db1_batch[j] / actual_batch_size);
            }
        }
        
        // Print epoch statistics
        float avg_loss = total_loss / images.size();
        cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << avg_loss << endl;
    }
    
    // Return trained weights
    return {w1, w2, b1, b2};
}

int argmax(const vector<float>& vec) {
    // Find the index of the maximum value
    int maxIndex = 0;
    float maxValue = vec[0];
    
    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] > maxValue) {
            maxValue = vec[i];
            maxIndex = i;
        }
    }
    
    return maxIndex;
}

float testTraining(const vector<vector<float>> &images, 
                   const vector<int> &labels,
                   TrainedWeights &weights) {
    int correct = 0;
    int total = images.size();
    
    cout << "\nTesting on " << total << " images..." << endl;
    
    for (int i = 0; i < total; i++) {
        // Forward pass only (no training)
        vector<float> hiddenZ = processLayer(weights.w1, weights.b1, images[i]);
        vector<float> hiddenA = relu(hiddenZ);
        vector<float> outputZ = processLayer(weights.w2, weights.b2, hiddenA);
        vector<float> y_hat = softmax(outputZ);
        
        // Get predicted digit (index of max probability)
        int predicted = argmax(y_hat);
        int actual = labels[i];
        
        if (predicted == actual) {
            correct++;
        }
        
        // Print progress every 10%
        if ((i + 1) % (total / 10) == 0) {
            float current_accuracy = (float)correct / (i + 1) * 100.0f;
            cout << "Progress: " << (i + 1) << "/" << total 
                 << " - Current Accuracy: " << current_accuracy << "%" << endl;
        }
    }
    
    float accuracy = (float)correct / total * 100.0f;
    cout << "\n=== TEST RESULTS ===" << endl;
    cout << "Correct: " << correct << "/" << total << endl;
    cout << "Accuracy: " << accuracy << "%" << endl;
    
    return accuracy;
}

void saveWeights(const string& prefix, const TrainedWeights &weights) {
    cout << "\nSaving weights to binary files..." << endl;
    
    // Get dimensions
    uint32_t w1_rows = weights.w1.size();
    uint32_t w1_cols = weights.w1[0].size();
    uint32_t w2_rows = weights.w2.size();
    uint32_t w2_cols = weights.w2[0].size();
    uint32_t b1_size = weights.b1.size();
    uint32_t b2_size = weights.b2.size();
    
    // Save W1
    {
        ofstream file(prefix + "_W1.bin", ios::binary);
        if (!file.is_open()) throw runtime_error("Failed to open W1 file");
        
        // Write dimensions (rows, cols)
        file.write((char*)&w1_rows, sizeof(w1_rows));
        file.write((char*)&w1_cols, sizeof(w1_cols));
        
        // Write data in row-major order
        for (int i = 0; i < w1_rows; i++) {
            for (int j = 0; j < w1_cols; j++) {
                float value = weights.w1[i][j];
                file.write((char*)&value, sizeof(value));
            }
        }
        file.close();
        cout << "  ✓ " << prefix << "_W1.bin: " << w1_rows << " × " << w1_cols 
             << " (" << (8 + w1_rows * w1_cols * 4) << " bytes)" << endl;
    }
    
    // Save b1
    {
        ofstream file(prefix + "_b1.bin", ios::binary);
        if (!file.is_open()) throw runtime_error("Failed to open b1 file");
        
        // Write size
        file.write((char*)&b1_size, sizeof(b1_size));
        
        // Write data
        for (int i = 0; i < b1_size; i++) {
            float value = weights.b1[i];
            file.write((char*)&value, sizeof(value));
        }
        file.close();
        cout << "  ✓ " << prefix << "_b1.bin: " << b1_size 
             << " values (" << (4 + b1_size * 4) << " bytes)" << endl;
    }
    
    // Save W2
    {
        ofstream file(prefix + "_W2.bin", ios::binary);
        if (!file.is_open()) throw runtime_error("Failed to open W2 file");
        
        // Write dimensions (rows, cols)
        file.write((char*)&w2_rows, sizeof(w2_rows));
        file.write((char*)&w2_cols, sizeof(w2_cols));
        
        // Write data in row-major order
        for (int i = 0; i < w2_rows; i++) {
            for (int j = 0; j < w2_cols; j++) {
                float value = weights.w2[i][j];
                file.write((char*)&value, sizeof(value));
            }
        }
        file.close();
        cout << "  ✓ " << prefix << "_W2.bin: " << w2_rows << " × " << w2_cols 
             << " (" << (8 + w2_rows * w2_cols * 4) << " bytes)" << endl;
    }
    
    // Save b2
    {
        ofstream file(prefix + "_b2.bin", ios::binary);
        if (!file.is_open()) throw runtime_error("Failed to open b2 file");
        
        // Write size
        file.write((char*)&b2_size, sizeof(b2_size));
        
        // Write data
        for (int i = 0; i < b2_size; i++) {
            float value = weights.b2[i];
            file.write((char*)&value, sizeof(value));
        }
        file.close();
        cout << "  ✓ " << prefix << "_b2.bin: " << b2_size 
             << " values (" << (4 + b2_size * 4) << " bytes)" << endl;
    }
    
    cout << "\nAll weights saved successfully!" << endl;
}

int main() {
    // Load training data
    cout << "=== LOADING TRAINING DATA ===" << endl;
    ifstream trainLabelFile("../train-labels.idx1-ubyte", ios::binary);
    ifstream trainImageFile("../train-images.idx3-ubyte", ios::binary);
    const vector<int> trainLabels = loadMNISTLabels(trainLabelFile);
    const vector<vector<float>> trainImages = loadMNISTImages(trainImageFile);
    trainLabelFile.close();
    trainImageFile.close();
    
    // Train the model
    cout << "\n=== TRAINING ===" << endl;
    TrainedWeights weights = runTraining(trainImages, trainLabels, 128, 10, 0.1f, 32, "MNIST_weights");
    
    // Load test data
    cout << "\n=== LOADING TEST DATA ===" << endl;
    ifstream testLabelFile("../t10k-labels.idx1-ubyte", ios::binary);
    ifstream testImageFile("../t10k-images.idx3-ubyte", ios::binary);
    const vector<int> testLabels = loadMNISTLabels(testLabelFile);
    const vector<vector<float>> testImages = loadMNISTImages(testImageFile);
    testLabelFile.close();
    testImageFile.close();
    
    // Test the model
    testTraining(testImages, testLabels, weights);
    
    // Save trained weights
    saveWeights("mnist_weights", weights);
    
    return 0;
}