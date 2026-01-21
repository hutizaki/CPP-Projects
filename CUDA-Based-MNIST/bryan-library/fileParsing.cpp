#include <fstream>
#include <vector>
#include <iostream>

using namespace std;

namespace fp {

int readBytesBE(ifstream& file, int byteSize) {
    vector<char> buffer(byteSize);

    file.read(buffer.data(), byteSize);
    int x = 0;
    for (int i = 0; i < byteSize; i++) {
        x = (x << 8) | (int) (unsigned char) buffer[i];
    }

    return x;
}

int readBytesLE(ifstream& file, int byteSize) {
    vector<char> buffer(byteSize);

    file.read(buffer.data(), byteSize);
    int x = 0;
    for (int i = 0; i < byteSize; i++) {
        x |= (int) (unsigned char) buffer[i] << (i * 8);
    }
    return x;
}

const vector<int> loadMNISTLabels(ifstream& file) {
    int labelMagicNum = 2049;
    int magicNum = readBytesBE(file, 4);
    if (labelMagicNum != magicNum) throw runtime_error("Magic numbers do not match! Incorrect file");
    int numLabels = readBytesBE(file, 4);
    vector<int> labelArr;
    cout << "Loading labels..." << endl;
    int tenth = numLabels/10;
    int percent = 0;
    for (int i = 0; i < numLabels; i++) {
        if (i % tenth == 0) {
            percent += 10;
            cout << percent << "% complete" << endl;
        }
        unsigned char byte;
        file.read((char*)&byte, 1);
        labelArr.push_back(byte);
    }
    cout << "Labels have been fully loaded" << endl;
    return labelArr;
}

const vector<vector<float>> loadMNISTImages(ifstream& file) {
    int imageMagicNum = 2051;
    int magicNum = readBytesBE(file, 4);
    if (imageMagicNum != magicNum) throw runtime_error("Magic numbers do not match! Incorrect file");

    int numImages = readBytesBE(file, 4);
    int numRows = readBytesBE(file, 4);
    int numCols = readBytesBE(file, 4);
    int tenth = numImages / 10;
    int percent = 0;
    vector<vector<float>> imageArr;
    vector<float> image;
    cout << "Loading images..." << endl;
    for (int i = 0; i < numImages; i++) {
        if (i % tenth == 0) {
            percent += 10;
            cout << percent << "% complete" << endl;
        }
        for (int j = 0; j < numRows * numCols; j++) {
            image.push_back(readBytesBE(file,1) / 255.0f);
        }
        imageArr.push_back(image);
        image.clear();
    }
    cout << "Images have been fully loaded" << endl;
    return imageArr;
}

}