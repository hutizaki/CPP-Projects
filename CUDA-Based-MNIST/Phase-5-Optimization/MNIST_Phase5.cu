#include "../bryan-library/machineLearning.h"
#include "../bryan-library/cudaMachineLearning.h"
#include "../bryan-library/cudaMatrixMath.h"
#include "../bryan-library/fileParsing.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

using namespace ml;
using namespace std;
using namespace fp;

struct TrainedWeights {
    vector<vector<float>> w1;
    vector<vector<float>> w2;
    vector<float> b1;
    vector<float> b2;
};

// Full GPU training - forward and backward pass entirely on GPU
TrainedWeights runTrainingGPU(const vector<vector<float>> &images, const vector<int> labels, 
                              int numNeurons, int epochs, float learning_rate, int batch_size) {
    int inputLength = images[0].size();
    
    // Try to load initial weights from Phase 5, otherwise generate random
    vector<vector<float>> w1, w2;
    vector<float> b1, b2;
    
    string initial_weights_path = "initial_weights";
    ifstream testFile(initial_weights_path + "_W1.bin", ios::binary);
    if (testFile.is_open()) {
        testFile.close();
        cout << "Loading initial weights for fair comparison..." << endl;
        // Load weights (same format as Phase 4)
        uint32_t w1_rows, w1_cols;
        ifstream w1File(initial_weights_path + "_W1.bin", ios::binary);
        w1File.read(reinterpret_cast<char*>(&w1_rows), sizeof(w1_rows));
        w1File.read(reinterpret_cast<char*>(&w1_cols), sizeof(w1_cols));
        w1.resize(w1_rows);
        for (auto& row : w1) {
            row.resize(w1_cols);
            w1File.read(reinterpret_cast<char*>(row.data()), w1_cols * sizeof(float));
        }
        w1File.close();
        
        uint32_t w2_rows, w2_cols;
        ifstream w2File(initial_weights_path + "_W2.bin", ios::binary);
        w2File.read(reinterpret_cast<char*>(&w2_rows), sizeof(w2_rows));
        w2File.read(reinterpret_cast<char*>(&w2_cols), sizeof(w2_cols));
        w2.resize(w2_rows);
        for (auto& row : w2) {
            row.resize(w2_cols);
            w2File.read(reinterpret_cast<char*>(row.data()), w2_cols * sizeof(float));
        }
        w2File.close();
        
        uint32_t b1_size;
        ifstream b1File(initial_weights_path + "_b1.bin", ios::binary);
        b1File.read(reinterpret_cast<char*>(&b1_size), sizeof(b1_size));
        b1.resize(b1_size);
        b1File.read(reinterpret_cast<char*>(b1.data()), b1_size * sizeof(float));
        b1File.close();
        
        uint32_t b2_size;
        ifstream b2File(initial_weights_path + "_b2.bin", ios::binary);
        b2File.read(reinterpret_cast<char*>(&b2_size), sizeof(b2_size));
        b2.resize(b2_size);
        b2File.read(reinterpret_cast<char*>(b2.data()), b2_size * sizeof(float));
        b2File.close();
    } else {
        // Initialize weights using library functions
        w1 = ml::generateRandMatrix(numNeurons, inputLength);
        w2 = ml::generateRandMatrix(10, numNeurons);
        b1 = ml::generateRandVector(numNeurons);
        b2 = ml::generateRandVector(10);
    }

    // Create GPU weights manager - keeps weights on GPU
    cudaMM::GPUWeights gpu_weights(w1, w2, b1, b2);

    // Allocate GPU memory for batch processing (reused each batch)
    float *d_batch_input, *d_hiddenZ, *d_hiddenA, *d_outputZ, *d_y_hat;
    int *d_trueLabels;
    float *d_dW1, *d_db1, *d_dW2, *d_db2;
    int max_batch_size = batch_size;
    
    CUDA_CHECK(cudaMalloc(&d_batch_input, max_batch_size * inputLength * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hiddenZ, max_batch_size * numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hiddenA, max_batch_size * numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputZ, max_batch_size * 10 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_hat, max_batch_size * 10 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trueLabels, max_batch_size * sizeof(int)));
    
    // Allocate gradient buffers (reused each batch)
    CUDA_CHECK(cudaMalloc(&d_dW1, numNeurons * inputLength * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW2, 10 * numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, 10 * sizeof(float)));

    // Host buffer for loss computation only
    vector<float> batch_y_hat(max_batch_size * 10);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        // Loop through batches
        for (int batch_start = 0; batch_start < images.size(); batch_start += batch_size) {
            int actual_batch_size = min(batch_size, (int)images.size() - batch_start);
            
            // === PREPARE BATCH DATA ===
            vector<float> batch_input(actual_batch_size * inputLength);
            vector<int> batch_labels(actual_batch_size);
            for (int i = 0; i < actual_batch_size; i++) {
                int idx = batch_start + i;
                batch_labels[i] = labels[idx];
                for (int j = 0; j < inputLength; j++) {
                    batch_input[i * inputLength + j] = images[idx][j];
                }
            }

            // Transfer batch to GPU
            CUDA_CHECK(cudaMemcpy(d_batch_input, batch_input.data(), 
                                  actual_batch_size * inputLength * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_trueLabels, batch_labels.data(),
                                  actual_batch_size * sizeof(int),
                                  cudaMemcpyHostToDevice));

            // === FORWARD PASS (GPU) ===
            cudaML::batchedForwardPass(gpu_weights, d_batch_input, actual_batch_size,
                                      d_hiddenZ, d_hiddenA, d_outputZ);
            CUDA_CHECK(cudaDeviceSynchronize());

            // === BACKWARD PASS (GPU) ===
            // Compute gradients and update weights entirely on GPU
            cudaML::batchedBackwardPass(gpu_weights, d_batch_input, d_trueLabels,
                                       d_hiddenZ, d_hiddenA, d_outputZ, d_y_hat,
                                       d_dW1, d_db1, d_dW2, d_db2,
                                       actual_batch_size, learning_rate);
            CUDA_CHECK(cudaDeviceSynchronize());

            // === COMPUTE LOSS (copy y_hat back for loss computation) ===
            CUDA_CHECK(cudaMemcpy(batch_y_hat.data(), d_y_hat,
                                  actual_batch_size * 10 * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            
            for (int i = 0; i < actual_batch_size; i++) {
                int idx = batch_start + i;
                vector<float> y_hat(10);
                for (int j = 0; j < 10; j++) {
                    y_hat[j] = batch_y_hat[i * 10 + j];
                }
                float loss = categoricalCrossEntropy(y_hat, labels[idx]);
                total_loss += loss;
            }
        }
        
        // Print epoch statistics
        float avg_loss = total_loss / images.size();
        cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << avg_loss << endl;
    }

    // Clean up GPU memory
    CUDA_CHECK(cudaFree(d_batch_input));
    CUDA_CHECK(cudaFree(d_hiddenZ));
    CUDA_CHECK(cudaFree(d_hiddenA));
    CUDA_CHECK(cudaFree(d_outputZ));
    CUDA_CHECK(cudaFree(d_y_hat));
    CUDA_CHECK(cudaFree(d_trueLabels));
    CUDA_CHECK(cudaFree(d_dW1));
    CUDA_CHECK(cudaFree(d_db1));
    CUDA_CHECK(cudaFree(d_dW2));
    CUDA_CHECK(cudaFree(d_db2));

    // Copy final weights back from GPU
    gpu_weights.copyToHost(w1, w2, b1, b2);
    
    // Return trained weights
    return {w1, w2, b1, b2};
}

int argmax(const vector<float>& vec) {
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
        // Forward pass only (no training) - using CPU for now
        vector<float> hiddenZ = processLayer(weights.w1, weights.b1, images[i]);
        vector<float> hiddenA = relu(hiddenZ);
        vector<float> outputZ = processLayer(weights.w2, weights.b2, hiddenA);
        vector<float> y_hat = softmax(outputZ);
        
        // Get predicted digit (index of max probability)
        int predicted = argmax(y_hat);
        
        if (predicted == labels[i]) {
            correct++;
        }
    }
    
    float accuracy = (float)correct / total * 100.0f;
    cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "% (" 
         << correct << "/" << total << ")" << endl;
    
    return accuracy;
}

void saveWeights(const string& filename, const TrainedWeights& weights) {
    ofstream w1File(filename + "_W1.bin", ios::binary);
    ofstream w2File(filename + "_W2.bin", ios::binary);
    ofstream b1File(filename + "_b1.bin", ios::binary);
    ofstream b2File(filename + "_b2.bin", ios::binary);
    
    // Save W1
    for (const auto& row : weights.w1) {
        w1File.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    
    // Save W2
    for (const auto& row : weights.w2) {
        w2File.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    
    // Save b1
    b1File.write(reinterpret_cast<const char*>(weights.b1.data()), weights.b1.size() * sizeof(float));
    
    // Save b2
    b2File.write(reinterpret_cast<const char*>(weights.b2.data()), weights.b2.size() * sizeof(float));
    
    w1File.close();
    w2File.close();
    b1File.close();
    b2File.close();
    
    cout << "Weights saved to " << filename << "_*.bin" << endl;
}

int main() {
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        cerr << "ERROR: No CUDA devices found or CUDA not available!" << endl;
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "=== CUDA DEVICE INFO ===" << endl;
    cout << "Device: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    cout << endl;
    
    // Load training data
    cout << "=== LOADING TRAINING DATA ===" << endl;
    ifstream trainLabelFile("../train-labels.idx1-ubyte", ios::binary);
    ifstream trainImageFile("../train-images.idx3-ubyte", ios::binary);
    const vector<int> trainLabels = loadMNISTLabels(trainLabelFile);
    const vector<vector<float>> trainImages = loadMNISTImages(trainImageFile);
    trainLabelFile.close();
    trainImageFile.close();
    
    // Training parameters
    int numNeurons = 128;
    int epochs = 10;
    float learning_rate = 0.1f;
    int batch_size = 32;
    
    // === GPU TRAINING ===
    cout << "\n=== GPU TRAINING (Phase 5) ===" << endl;
    cout << "Forward and backward pass entirely on GPU!" << endl;
    auto start_gpu = chrono::high_resolution_clock::now();
    TrainedWeights weights_gpu = runTrainingGPU(trainImages, trainLabels, numNeurons, epochs, learning_rate, batch_size);
    auto end_gpu = chrono::high_resolution_clock::now();
    auto duration_gpu = chrono::duration_cast<chrono::milliseconds>(end_gpu - start_gpu);
    
    cout << "\n=== TRAINING TIME ===" << endl;
    cout << "GPU Training Time: " << duration_gpu.count() << " ms (" 
         << fixed << setprecision(2) << duration_gpu.count() / 1000.0 << " seconds)" << endl;
    cout << endl;
    
    // Load test data
    cout << "=== LOADING TEST DATA ===" << endl;
    ifstream testLabelFile("../t10k-labels.idx1-ubyte", ios::binary);
    ifstream testImageFile("../t10k-images.idx3-ubyte", ios::binary);
    const vector<int> testLabels = loadMNISTLabels(testLabelFile);
    const vector<vector<float>> testImages = loadMNISTImages(testImageFile);
    testLabelFile.close();
    testImageFile.close();
    
    // === TEST GPU MODEL ===
    cout << "\n=== TESTING GPU MODEL ===" << endl;
    float accuracy_gpu = testTraining(testImages, testLabels, weights_gpu);
    
    cout << "\n=== FINAL RESULTS ===" << endl;
    cout << "GPU Accuracy: " << fixed << setprecision(2) << accuracy_gpu << "%" << endl;
    cout << "Training Time: " << duration_gpu.count() / 1000.0 << " seconds" << endl;
    cout << endl;
    
    // Save trained weights
    saveWeights("gpu_weights", weights_gpu);
    
    return 0;
}
