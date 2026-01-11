# 📦 Binary Weights File Format

This document describes the binary file format used to store trained neural network weights for the MNIST classifier.

---

## 📁 Files Generated

After training, the program saves **4 separate binary files**:

1. **`mnist_weights_W1.bin`** - Hidden layer weights (128 × 784)
2. **`mnist_weights_b1.bin`** - Hidden layer biases (128)
3. **`mnist_weights_W2.bin`** - Output layer weights (10 × 128)
4. **`mnist_weights_b2.bin`** - Output layer biases (10)

---

## 🔢 Data Types

All values are stored as:
- **Dimensions:** `uint32_t` (4 bytes, unsigned 32-bit integer)
- **Weights/Biases:** `float` (4 bytes, 32-bit floating point)
- **Byte Order:** Little-endian (standard for x86/ARM)

---

## 📋 File Format Specifications

### 1. Weight Matrix Files (`_W1.bin`, `_W2.bin`)

**Structure:**
```
Bytes 0-3:   rows (uint32_t)
Bytes 4-7:   cols (uint32_t)
Bytes 8+:    weight values (float[rows × cols])
```

**Storage Order:** Row-major
- Values are stored row by row
- Element at position [i][j] is at byte: `8 + (i × cols + j) × 4`

**Example for W1 (128 × 784):**
```
Offset  | Content
--------|--------------------------------------------------
0-3     | 128 (number of rows)
4-7     | 784 (number of columns)
8-11    | W1[0][0] (first weight of first neuron)
12-15   | W1[0][1] (second weight of first neuron)
...     | ...
3144-3147| W1[0][783] (last weight of first neuron)
3148-3151| W1[1][0] (first weight of second neuron)
...     | ...
401415  | W1[127][783] (last weight of last neuron)
```

**File Size:**
- W1: 8 + (128 × 784 × 4) = **401,416 bytes** (~392 KB)
- W2: 8 + (10 × 128 × 4) = **5,128 bytes** (~5 KB)

---

### 2. Bias Vector Files (`_b1.bin`, `_b2.bin`)

**Structure:**
```
Bytes 0-3:   size (uint32_t)
Bytes 4+:    bias values (float[size])
```

**Storage Order:** Sequential
- Values are stored in order from index 0 to size-1
- Element at position [i] is at byte: `4 + i × 4`

**Example for b1 (128):**
```
Offset  | Content
--------|--------------------------------------------------
0-3     | 128 (number of biases)
4-7     | b1[0] (bias for first hidden neuron)
8-11    | b1[1] (bias for second hidden neuron)
...     | ...
512-515 | b1[127] (bias for last hidden neuron)
```

**File Size:**
- b1: 4 + (128 × 4) = **516 bytes**
- b2: 4 + (10 × 4) = **44 bytes**

---

## 💻 Reading the Files in C++

### Read Weight Matrix

```cpp
vector<vector<float>> loadWeightMatrix(const string& filename) {
    ifstream file(filename, ios::binary);
    
    // Read dimensions
    uint32_t rows, cols;
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));
    
    // Read weights
    vector<vector<float>> weights(rows, vector<float>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file.read((char*)&weights[i][j], sizeof(float));
        }
    }
    
    file.close();
    return weights;
}
```

### Read Bias Vector

```cpp
vector<float> loadBiasVector(const string& filename) {
    ifstream file(filename, ios::binary);
    
    // Read size
    uint32_t size;
    file.read((char*)&size, sizeof(size));
    
    // Read biases
    vector<float> biases(size);
    for (int i = 0; i < size; i++) {
        file.read((char*)&biases[i], sizeof(float));
    }
    
    file.close();
    return biases;
}
```

---

## 🧮 Network Architecture

The saved weights represent this network:

```
Input Layer (784 neurons)
    ↓
    W1 (128 × 784) + b1 (128)
    ↓
Hidden Layer (128 neurons, ReLU activation)
    ↓
    W2 (10 × 128) + b2 (10)
    ↓
Output Layer (10 neurons, Softmax activation)
```

**Forward Pass:**
```
hiddenZ = W1 × input + b1
hiddenA = ReLU(hiddenZ)
outputZ = W2 × hiddenA + b2
output = Softmax(outputZ)
```

---

## 📊 Total Storage Size

| File | Dimensions | Size |
|------|------------|------|
| `mnist_weights_W1.bin` | 128 × 784 | ~392 KB |
| `mnist_weights_b1.bin` | 128 | 516 bytes |
| `mnist_weights_W2.bin` | 10 × 128 | ~5 KB |
| `mnist_weights_b2.bin` | 10 | 44 bytes |
| **TOTAL** | | **~397 KB** |

---

## 🔍 Verifying Files

### Check File Sizes

```bash
ls -lh mnist_weights_*.bin
```

**Expected output:**
```
-rw-r--r--  1 user  staff   392K  mnist_weights_W1.bin
-rw-r--r--  1 user  staff   516B  mnist_weights_b1.bin
-rw-r--r--  1 user  staff   5.0K  mnist_weights_W2.bin
-rw-r--r--  1 user  staff    44B  mnist_weights_b2.bin
```

### Inspect Binary Content

```bash
# View first 16 bytes in hex
hexdump -C mnist_weights_W1.bin | head -n 2
```

**Expected output (example):**
```
00000000  80 00 00 00 10 03 00 00  3f 12 a4 be 8c 9f 1a 3f  |........?......?|
          ^-rows=128  ^-cols=784  ^-first weight value
```

---

## 🚀 Usage in CUDA

These binary files are designed to be easily loaded into GPU memory:

```cpp
// Load on CPU
vector<vector<float>> W1_cpu = loadWeightMatrix("mnist_weights_W1.bin");

// Flatten for GPU
float* W1_flat = flatten(W1_cpu);  // 128 × 784 = 100,352 floats

// Copy to GPU
float* W1_gpu;
cudaMalloc(&W1_gpu, 128 * 784 * sizeof(float));
cudaMemcpy(W1_gpu, W1_flat, 128 * 784 * sizeof(float), cudaMemcpyHostToDevice);
```

---

## 📝 Notes

- **Row-major order** matches C++ `vector<vector<float>>` layout
- **No compression** - raw float values for fast loading
- **No checksums** - add validation if needed for production
- **Platform-specific** - assumes little-endian architecture (x86, ARM)
- **Version 1.0** - format may be extended in future phases

---

## ✅ Validation

After saving, the program prints:
```
Saving weights to binary files...
  ✓ mnist_weights_W1.bin: 128 × 784 (401416 bytes)
  ✓ mnist_weights_b1.bin: 128 values (516 bytes)
  ✓ mnist_weights_W2.bin: 10 × 128 (5128 bytes)
  ✓ mnist_weights_b2.bin: 10 values (44 bytes)

All weights saved successfully!
```

This confirms all files were written correctly! 🎉

