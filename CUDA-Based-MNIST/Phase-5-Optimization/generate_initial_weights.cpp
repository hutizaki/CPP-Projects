#include "../bryan-library/machineLearning.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

using namespace ml;
using namespace std;

void saveMatrix(const string& filename, const vector<vector<float>>& matrix) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }
    
    uint32_t rows = matrix.size();
    uint32_t cols = matrix[0].size();
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    // Write data in row-major order
    for (const auto& row : matrix) {
        file.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(float));
    }
    
    file.close();
    cout << "Saved " << filename << ": " << rows << "x" << cols << endl;
}

void saveVector(const string& filename, const vector<float>& vec) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }
    
    uint32_t size = vec.size();
    
    // Write size
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    // Write data
    file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
    
    file.close();
    cout << "Saved " << filename << ": " << size << " elements" << endl;
}

int main(int argc, char* argv[]) {
    // Set random seed for reproducibility
    srand(42);  // Fixed seed for reproducible weights
    
    int numNeurons = 128;
    int inputLength = 784;
    
    if (argc > 1) {
        numNeurons = atoi(argv[1]);
    }
    if (argc > 2) {
        inputLength = atoi(argv[2]);
    }
    
    cout << "Generating initial weights..." << endl;
    cout << "Architecture: " << inputLength << " -> " << numNeurons << " -> 10" << endl;
    cout << "Random seed: 42 (fixed for reproducibility)" << endl << endl;
    
    // Generate weights matching Phase 4
    vector<vector<float>> w1 = generateRandMatrix(numNeurons, inputLength);
    vector<vector<float>> w2 = generateRandMatrix(10, numNeurons);
    vector<float> b1 = generateRandVector(numNeurons);
    vector<float> b2 = generateRandVector(10);
    
    // Save weights
    string prefix = "initial_weights";
    saveMatrix(prefix + "_W1.bin", w1);
    saveMatrix(prefix + "_W2.bin", w2);
    saveVector(prefix + "_b1.bin", b1);
    saveVector(prefix + "_b2.bin", b2);
    
    cout << "\nAll initial weights saved successfully!" << endl;
    cout << "Files saved in current directory." << endl;
    
    return 0;
}
