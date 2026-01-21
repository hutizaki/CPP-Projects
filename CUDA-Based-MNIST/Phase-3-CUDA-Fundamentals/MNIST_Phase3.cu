#include "../bryan-library/matrixMath.h"
#include "../bryan-library/machineLearning.h"
#include "../bryan-library/cudaMachineLearning.h"
#include "../bryan-library/cudaMatrixMath.h"
#include "../bryan-library/fileParsing.h"
#include <cstdint>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cmath>

// ============================================================================
// CONFIGURATION: Set to true to enable CPU training for comparison
// ============================================================================
const bool usingCPU = true;  // Change to true to run CPU training and compare with GPU
// ============================================================================

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

TrainedWeights runTrainingCPU(const vector<vector<float>> &images, const vector<int> labels, int numNeurons, int epochs, float learning_rate, int batch_size, string function_name,
                              const vector<vector<float>>* initial_w1 = nullptr,
                              const vector<vector<float>>* initial_w2 = nullptr,
                              const vector<float>* initial_b1 = nullptr,
                              const vector<float>* initial_b2 = nullptr)  {
    int inputLength = images[0].size();
    vector<vector<float>> w1, w2;
    vector<float> b1, b2;
    
    if (initial_w1 && initial_w2 && initial_b1 && initial_b2) {
        // Use provided initial weights
        w1 = *initial_w1;
        w2 = *initial_w2;
        b1 = *initial_b1;
        b2 = *initial_b2;
    } else {
        // Generate random weights
        w1 = generateRandMatrix(numNeurons, inputLength);
        w2 = generateRandMatrix(10, numNeurons);
        b1 = generateRandVector(numNeurons);
        b2 = generateRandVector(10);
    }

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
            
            // Process each sample in the batch (CPU)
            for (int i = 0; i < actual_batch_size; i++) {
                int idx = batch_start + i;
                
                // === FORWARD PASS (CPU) ===
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
            
            // === UPDATE WEIGHTS ===
            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < numNeurons; k++) {
                    w2[j][k] -= learning_rate * (dW2_batch[j][k] / actual_batch_size);
                }
                b2[j] -= learning_rate * (db2_batch[j] / actual_batch_size);
            }
            
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

TrainedWeights runTrainingGPU(const vector<vector<float>> &images, const vector<int> labels, int numNeurons, int epochs, float learning_rate, int batch_size, string function_name,
                              const vector<vector<float>>* initial_w1 = nullptr,
                              const vector<vector<float>>* initial_w2 = nullptr,
                              const vector<float>* initial_b1 = nullptr,
                              const vector<float>* initial_b2 = nullptr)  {
    int inputLength = images[0].size();
    vector<vector<float>> w1, w2;
    vector<float> b1, b2;
    
    if (initial_w1 && initial_w2 && initial_b1 && initial_b2) {
        // Use provided initial weights
        w1 = *initial_w1;
        w2 = *initial_w2;
        b1 = *initial_b1;
        b2 = *initial_b2;
    } else {
        // Generate random weights
        w1 = generateRandMatrix(numNeurons, inputLength);
        w2 = generateRandMatrix(10, numNeurons);
        b1 = generateRandVector(numNeurons);
        b2 = generateRandVector(10);
    }

    // Create GPU weights manager - keeps weights on GPU
    cudaMM::GPUWeights gpu_weights(w1, w2, b1, b2);

    // Allocate GPU memory for batch processing (reused each batch)
    float *d_batch_input, *d_hiddenZ, *d_hiddenA, *d_outputZ;
    int max_batch_size = batch_size;
    CUDA_CHECK(cudaMalloc(&d_batch_input, max_batch_size * inputLength * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hiddenZ, max_batch_size * numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hiddenA, max_batch_size * numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputZ, max_batch_size * 10 * sizeof(float)));

    // Host buffers for copying results back
    vector<float> batch_hiddenZ(max_batch_size * numNeurons);
    vector<float> batch_hiddenA(max_batch_size * numNeurons);
    vector<float> batch_outputZ(max_batch_size * 10);

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
            
            // === BATCHED GPU FORWARD PASS ===
            // Prepare batch input on host
            vector<float> batch_input(actual_batch_size * inputLength);
            for (int i = 0; i < actual_batch_size; i++) {
                int idx = batch_start + i;
                for (int j = 0; j < inputLength; j++) {
                    batch_input[i * inputLength + j] = images[idx][j];
                }
            }

            // Transfer batch to GPU
            CUDA_CHECK(cudaMemcpy(d_batch_input, batch_input.data(), 
                                  actual_batch_size * inputLength * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Process entire batch on GPU
            cudaML::batchedForwardPass(gpu_weights, d_batch_input, actual_batch_size,
                                      d_hiddenZ, d_hiddenA, d_outputZ);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy results back to host
            CUDA_CHECK(cudaMemcpy(batch_hiddenZ.data(), d_hiddenZ,
                                  actual_batch_size * numNeurons * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(batch_hiddenA.data(), d_hiddenA,
                                  actual_batch_size * numNeurons * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(batch_outputZ.data(), d_outputZ,
                                  actual_batch_size * 10 * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // === BACKWARD PASS (CPU for now) ===
            for (int i = 0; i < actual_batch_size; i++) {
                int idx = batch_start + i;
                
                // Extract results for this sample
                vector<float> hiddenZ(numNeurons);
                vector<float> hiddenA(numNeurons);
                vector<float> outputZ(10);
                for (int j = 0; j < numNeurons; j++) {
                    hiddenZ[j] = batch_hiddenZ[i * numNeurons + j];
                    hiddenA[j] = batch_hiddenA[i * numNeurons + j];
                }
                for (int j = 0; j < 10; j++) {
                    outputZ[j] = batch_outputZ[i * 10 + j];
                }

                // Softmax and loss (CPU)
                vector<float> y_hat = softmax(outputZ);
                float loss = categoricalCrossEntropy(y_hat, labels[idx]);
                total_loss += loss;
                
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
            
            // === UPDATE WEIGHTS (CPU - copy back to GPU next iteration) ===
            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < numNeurons; k++) {
                    w2[j][k] -= learning_rate * (dW2_batch[j][k] / actual_batch_size);
                }
                b2[j] -= learning_rate * (db2_batch[j] / actual_batch_size);
            }
            
            for (int j = 0; j < numNeurons; j++) {
                for (int k = 0; k < inputLength; k++) {
                    w1[j][k] -= learning_rate * (dW1_batch[j][k] / actual_batch_size);
                }
                b1[j] -= learning_rate * (db1_batch[j] / actual_batch_size);
            }

            // Update GPU weights for next batch (copy updated weights back to GPU)
            std::vector<float> w1_flat = cudaMM::rowMajorMatrix(w1);
            std::vector<float> w2_flat = cudaMM::rowMajorMatrix(w2);
            CUDA_CHECK(cudaMemcpy(gpu_weights.d_w1, w1_flat.data(), 
                                  gpu_weights.w1_rows * gpu_weights.w1_cols * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gpu_weights.d_w2, w2_flat.data(), 
                                  gpu_weights.w2_rows * gpu_weights.w2_cols * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gpu_weights.d_b1, b1.data(), 
                                  gpu_weights.b1_size * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gpu_weights.d_b2, b2.data(), 
                                  gpu_weights.b2_size * sizeof(float),
                                  cudaMemcpyHostToDevice));
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

    // Copy final weights back from GPU
    gpu_weights.copyToHost(w1, w2, b1, b2);
    
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
        // Using CPU version for now - CUDA per-sample is too slow due to overhead
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

// Helper function to compare weights
float compareWeights(const TrainedWeights& w1, const TrainedWeights& w2) {
    float total_diff = 0.0f;
    int total_elements = 0;
    
    // Compare W1
    for (size_t i = 0; i < w1.w1.size(); i++) {
        for (size_t j = 0; j < w1.w1[i].size(); j++) {
            total_diff += abs(w1.w1[i][j] - w2.w1[i][j]);
            total_elements++;
        }
    }
    
    // Compare W2
    for (size_t i = 0; i < w1.w2.size(); i++) {
        for (size_t j = 0; j < w1.w2[i].size(); j++) {
            total_diff += abs(w1.w2[i][j] - w2.w2[i][j]);
            total_elements++;
        }
    }
    
    // Compare b1
    for (size_t i = 0; i < w1.b1.size(); i++) {
        total_diff += abs(w1.b1[i] - w2.b1[i]);
        total_elements++;
    }
    
    // Compare b2
    for (size_t i = 0; i < w1.b2.size(); i++) {
        total_diff += abs(w1.b2[i] - w2.b2[i]);
        total_elements++;
    }
    
    return total_diff / total_elements;
}

int main() {
    // Check CUDA availability (always needed for GPU training)
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
    
    TrainedWeights weights_cpu, weights_gpu;
    chrono::milliseconds duration_cpu, duration_gpu;
    
    // Generate initial weights once (for fair comparison if using CPU)
    int inputLength = trainImages[0].size();
    vector<vector<float>> initial_w1, initial_w2;
    vector<float> initial_b1, initial_b2;
    
    if (usingCPU) {
        // Generate initial weights for fair comparison
        initial_w1 = generateRandMatrix(numNeurons, inputLength);
        initial_w2 = generateRandMatrix(10, numNeurons);
        initial_b1 = generateRandVector(numNeurons);
        initial_b2 = generateRandVector(10);
        
        // === CPU TRAINING ===
        cout << "\n=== CPU TRAINING ===" << endl;
        auto start_cpu = chrono::high_resolution_clock::now();
        weights_cpu = runTrainingCPU(trainImages, trainLabels, numNeurons, epochs, learning_rate, batch_size, "CPU",
                                     &initial_w1, &initial_w2, &initial_b1, &initial_b2);
        auto end_cpu = chrono::high_resolution_clock::now();
        duration_cpu = chrono::duration_cast<chrono::milliseconds>(end_cpu - start_cpu);
    }
    
    // === GPU TRAINING (always runs) ===
    cout << "\n=== GPU TRAINING ===" << endl;
    auto start_gpu = chrono::high_resolution_clock::now();
    if (usingCPU) {
        // Use same initial weights for fair comparison
        weights_gpu = runTrainingGPU(trainImages, trainLabels, numNeurons, epochs, learning_rate, batch_size, "GPU",
                                    &initial_w1, &initial_w2, &initial_b1, &initial_b2);
    } else {
        // Generate new random weights
        weights_gpu = runTrainingGPU(trainImages, trainLabels, numNeurons, epochs, learning_rate, batch_size, "GPU");
    }
    auto end_gpu = chrono::high_resolution_clock::now();
    duration_gpu = chrono::duration_cast<chrono::milliseconds>(end_gpu - start_gpu);
    
    // === TIMING ===
    if (usingCPU) {
        cout << "\n=== TIMING COMPARISON ===" << endl;
        cout << "CPU Training Time: " << duration_cpu.count() << " ms (" 
             << fixed << setprecision(2) << duration_cpu.count() / 1000.0 << " seconds)" << endl;
        cout << "GPU Training Time: " << duration_gpu.count() << " ms (" 
             << fixed << setprecision(2) << duration_gpu.count() / 1000.0 << " seconds)" << endl;
        double speedup = (double)duration_cpu.count() / duration_gpu.count();
        cout << "Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
    } else {
        cout << "\n=== TRAINING TIME ===" << endl;
        cout << "GPU Training Time: " << duration_gpu.count() << " ms (" 
             << fixed << setprecision(2) << duration_gpu.count() / 1000.0 << " seconds)" << endl;
    }
    cout << endl;
    
    // Load test data
    cout << "=== LOADING TEST DATA ===" << endl;
    ifstream testLabelFile("../t10k-labels.idx1-ubyte", ios::binary);
    ifstream testImageFile("../t10k-images.idx3-ubyte", ios::binary);
    const vector<int> testLabels = loadMNISTLabels(testLabelFile);
    const vector<vector<float>> testImages = loadMNISTImages(testImageFile);
    testLabelFile.close();
    testImageFile.close();
    
    if (usingCPU) {
        // === TEST CPU MODEL ===
        cout << "\n=== TESTING CPU MODEL ===" << endl;
        float accuracy_cpu = testTraining(testImages, testLabels, weights_cpu);
        
        // === TEST GPU MODEL ===
        cout << "\n=== TESTING GPU MODEL ===" << endl;
        float accuracy_gpu = testTraining(testImages, testLabels, weights_gpu);
        
        // === COMPARE RESULTS ===
        cout << "\n=== RESULTS COMPARISON ===" << endl;
        cout << "CPU Accuracy: " << fixed << setprecision(2) << accuracy_cpu << "%" << endl;
        cout << "GPU Accuracy: " << fixed << setprecision(2) << accuracy_gpu << "%" << endl;
        cout << "Accuracy Difference: " << fixed << setprecision(2) << abs(accuracy_cpu - accuracy_gpu) << "%" << endl;
        
        float avg_weight_diff = compareWeights(weights_cpu, weights_gpu);
        cout << "Average Weight Difference: " << scientific << setprecision(4) << avg_weight_diff << endl;
        
        if (avg_weight_diff < 0.01f && abs(accuracy_cpu - accuracy_gpu) < 1.0f) {
            cout << "✓ Results match closely - GPU implementation is correct!" << endl;
        } else {
            cout << "⚠ Warning: Results differ significantly - check implementation" << endl;
        }
        cout << endl;
    } else {
        // === TEST GPU MODEL ===
        cout << "\n=== TESTING GPU MODEL ===" << endl;
        float accuracy_gpu = testTraining(testImages, testLabels, weights_gpu);
        
        cout << "\n=== FINAL RESULTS ===" << endl;
        cout << "GPU Accuracy: " << fixed << setprecision(2) << accuracy_gpu << "%" << endl;
        cout << "Training Time: " << duration_gpu.count() / 1000.0 << " seconds" << endl;
        cout << endl;
    }
    
    // Save trained weights (using GPU version)
    saveWeights("mnist_weights", weights_gpu);
    
    return 0;
}