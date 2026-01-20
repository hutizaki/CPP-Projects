#ifndef FILEPARSING_H
#define FILEPARSING_H

#include <vector>
#include <fstream>

using namespace std;

namespace fp {
    int readBytesBE(ifstream& file, int byteSize);
    int readBytesLE(ifstream& file, int byteSize);
    vector<int> loadMNISTLabels(ifstream& file);
    vector<vector<int>> loadMNISTImages(ifstream& file);
}

#endif