#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <fstream>
#include <ios>
#include <iostream>
#include <sys/types.h>
#include <vector>
#include <sstream>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace sf;

uint WIDTH;
uint HEIGHT;

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

string intToHexPadded(int value, int width = 4) {
    stringstream ss;
    ss << hex << uppercase << setw(width) << setfill('0') << value;
    return ss.str();
}

int main()
{
    ifstream photo("sample_640x426.bmp", ios::binary);
    
    if (!photo) {
        cerr << "Failed to open file!" << endl;
        return 1;
    }
    
    char byte;
    int count = 0;

    string fileSignature = intToHexPadded(readBytesBE(photo, 2));

    if (fileSignature != "424D") {
        cerr << "File type isn't BMP!" << endl;
        return 1;
    }


    photo.seekg(10, ios::beg);

    int imageBeg = readBytesLE(photo, 4);

    photo.seekg(18, ios::beg);
    WIDTH = readBytesLE(photo, 4);
    HEIGHT = readBytesLE(photo, 4);
    photo.seekg(2, ios::cur);
    int BPP = readBytesLE(photo, 2);

    photo.seekg(imageBeg, ios::beg);

    cout << "Width: " << WIDTH << " Height: " << HEIGHT << " Bits Per Pixel: " << BPP << endl;

    Image image({WIDTH, HEIGHT}, Color::Black);

    for (uint y = HEIGHT - 1; y > 0; y--) {
        for (uint x = 0; x < WIDTH; x++) {
            u_char b = readBytesLE(photo, 1);
            u_char g = readBytesLE(photo, 1);
            u_char r = readBytesLE(photo, 1);

            cout << "Coord: (" << x << " ," << y << ")\n";
            image.setPixel({x, y}, Color(r, g, b));
        }
    }

    Texture t(image);
    Sprite s(t);

    RenderWindow window(VideoMode({WIDTH, HEIGHT}), "BMP Viewer");

    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<Event::Closed>())
            window.close();
        }

      window.clear();
      window.draw(s);
      window.display();
    }

    photo.close();
    return 0;
}