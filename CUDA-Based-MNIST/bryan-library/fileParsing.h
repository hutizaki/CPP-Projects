#ifndef FILEPARSING_H
#define FILEPARSING_H

#include <vector>
#include <fstream>

using namespace std;

namespace fp {
    int readBytesBE(ifstream& file, int byteSize);
    int readBytesLE(ifstream& file, int byteSize);
    const vector<int> loadMNISTLabels(ifstream& file);
    const vector<vector<float>> loadMNISTImages(ifstream& file);
}

#endif