#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    ifstream photo("sample_640x426.bmp", ios::binary);
    
    if (!photo) {
        cerr << "Failed to open file!" << endl;
        return 1;
    }
    
    // Read and print as hex
    char byte;
    int count = 0;
    while (photo.read(&byte, 1)) {
        cout << hex << setw(2) << setfill('0') 
             << (int)(unsigned char)byte << " ";
        
        // New line every 16 bytes (like a hex editor)
        if (++count % 16 == 0) cout << endl;
    }
    
    photo.close();
    return 0;
}