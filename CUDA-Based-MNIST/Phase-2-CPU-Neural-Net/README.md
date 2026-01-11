# 🖥️ **Phase 2: CPU Neural Network**

## Philosophy

**You cannot code what you don't understand.**

This phase teaches you how to translate neural network math into C++ code. You'll build each component step-by-step, understanding *why* before implementing *what*.

---

## 🎯 **Goal**

Build a neural network in C++ that learns to recognize handwritten digits from MNIST, achieving 90%+ accuracy.

**Time:** 7-10 days  
**Difficulty:** ⭐⭐⭐⭐

---

## 🤔 **Why This Phase Matters**

- You'll transform math (from Phase 1) into working code
- You'll build a reference implementation for verifying GPU code later
- You'll deeply understand how neural networks actually work
- No frameworks, no magic - just you and C++

---

## 📚 **What You'll Build**

A neural network with:
- **Input:** 784 pixels (28×28 image)
- **Hidden layer:** 128 neurons with ReLU
- **Output layer:** 10 neurons with Softmax (digits 0-9)

**The same concepts as your OR neuron, just scaled up!**

---

## 📋 **Function Checklist - What You Need to Build**

### ✅ Data Loading Functions (COMPLETED)

```cpp
int readBytes(ifstream& file, int byteSize) {
    // Reads 4 bytes from file and combines into int32_t
}

vector<int> loadMNISTLabels(ifstream& file) {
    // Loads all labels from MNIST labels file
}

vector<vector<float>> loadMNISTImages(ifstream& file) {
    // Loads all images from MNIST images file
    // Returns: vector of images, each image is 784 normalized floats
}
```

### ✅ Weight Initialization (COMPLETED)

```cpp
vector<vector<float>> buildWeightMatrix(int numRows, int numCols) {
    // Creates weight matrix with random values
}

// Already in machineLearning.h:
float randomWeight() {
    // Returns random weight for initialization
}
```

### 🔨 Bias Initialization (NEED TO BUILD)

**Critical: Bias Size Must Match Layer Output Size!**

```cpp
vector<float> initializeBias(int size) {
    // Creates bias vector initialized to small random values or zeros
    // Input: size (number of neurons in the layer)
    // Output: bias vector
}
```

**Weight and Bias Dimensions for MNIST Network:**

```
Network Architecture:
Input (784) → Hidden (128) → Output (10)

Weight matrices needed:
- W1: 128 × 784  (128 rows for 128 neurons, 784 cols for 784 inputs)
- W2: 10 × 128   (10 rows for 10 outputs, 128 cols for 128 hidden neurons)

Bias vectors needed:
- b1: 128 (one bias per hidden neuron)
- b2: 10  (one bias per output neuron)
```

**The Rules:**
- **Weight matrix size:** (output neurons) × (input size)
- **Bias vector size:** (output neurons) - one bias per neuron in THAT layer

```cpp
// Example initialization in runTraining():
vector<vector<float>> W1 = buildWeightMatrix(128, 784);  // 128 × 784
vector<float> b1 = initializeBias(128);                  // 128 biases

vector<vector<float>> W2 = buildWeightMatrix(10, 128);   // 10 × 128
vector<float> b2 = initializeBias(10);                   // 10 biases
```

**Why this size?**
- Each neuron has ONE bias value
- Hidden layer has 128 neurons → 128 biases
- Output layer has 10 neurons → 10 biases

**Common mistake:** Using input size for bias (❌ 784 biases for hidden layer)

### 🔨 Activation Functions (NEED TO BUILD)

```cpp
vector<float> ReLU(const vector<float>& z) {
    // Applies ReLU activation: max(0, z) element-wise
    // Input: z (pre-activation values)
    // Output: activated values
}

vector<float> softmax(const vector<float>& z) {
    // Converts logits to probabilities that sum to 1
    // Input: z (raw scores)
    // Output: probability distribution
}

// Note: sigmoid already exists in machineLearning.h for single values
```

### 🔨 Forward Pass Functions (NEED TO BUILD)

```cpp
vector<float> forwardHiddenLayer(const vector<vector<float>>& weights, 
                                  const vector<float>& bias, 
                                  const vector<float>& input) {
    // Computes: h = ReLU(W1 × input + b1)
    // Input: weights (128×784), bias (128), input (784)
    // Output: hidden activations (128)
}

vector<float> forwardOutputLayer(const vector<vector<float>>& weights, 
                                  const vector<float>& bias, 
                                  const vector<float>& hidden) {
    // Computes: output = softmax(W2 × hidden + b2)
    // Input: weights (10×128), bias (10), hidden (128)
    // Output: probabilities (10)
}
```

### 🔨 Loss Function (NEED TO BUILD)

```cpp
float categoricalCrossEntropy(const vector<float>& predictions, int trueLabel) {
    // Computes loss: -log(predictions[trueLabel])
    // Input: predictions (10 probabilities), trueLabel (0-9)
    // Output: loss value
}
```

### 🔨 Backward Pass Functions (NEED TO BUILD)

```cpp
vector<float> softmaxGradient(const vector<float>& output, int trueLabel) {
    // Computes gradient of loss w.r.t. output layer pre-activation
    // For softmax + cross-entropy: dL/dz = output - one_hot(trueLabel)
    // Input: output (10 probabilities), trueLabel (0-9)
    // Output: gradient (10)
}

vector<float> ReLUGradient(const vector<float>& z, const vector<float>& gradOutput) {
    // Computes gradient through ReLU
    // ReLU'(z) = 1 if z > 0, else 0
    // Input: z (pre-activation), gradOutput (gradient from next layer)
    // Output: gradient (element-wise product)
}

void updateWeights(vector<vector<float>>& weights, 
                   vector<float>& bias,
                   const vector<float>& gradWeights,
                   const vector<float>& gradBias,
                   float learningRate) {
    // Updates weights and biases using gradients
    // weights -= learningRate * gradWeights
    // bias -= learningRate * gradBias
}
```

### 🔨 Gradient Computation (NEED TO BUILD)

```cpp
vector<vector<float>> computeWeightGradient(const vector<float>& input, 
                                             const vector<float>& gradOutput) {
    // Computes dL/dW = gradOutput × input^T
    // Input: input (n), gradOutput (m)
    // Output: gradient matrix (m×n)
}

vector<float> computeInputGradient(const vector<vector<float>>& weights, 
                                    const vector<float>& gradOutput) {
    // Computes dL/dinput = W^T × gradOutput
    // Input: weights (m×n), gradOutput (m)
    // Output: gradient (n)
}
```

### 🔨 Training Loop (NEED TO BUILD)

```cpp
void runTraining(const vector<vector<float>>& images, 
                 const vector<int>& labels, 
                 int numHiddenNeurons, 
                 int epochs, 
                 float learningRate, 
                 string outputFileName) {
    // Main training loop:
    // 1. Initialize weights (W1, b1, W2, b2)
    // 2. For each epoch:
    //    - For each training sample:
    //      a. Forward pass
    //      b. Compute loss
    //      c. Backward pass
    //      d. Update weights
    //    - Print epoch loss
    // 3. Save final weights to file
}
```

### 🔨 Testing/Evaluation (NEED TO BUILD)

```cpp
float testAccuracy(const vector<vector<float>>& images, 
                   const vector<int>& labels,
                   const vector<vector<float>>& W1,
                   const vector<float>& b1,
                   const vector<vector<float>>& W2,
                   const vector<float>& b2) {
    // Evaluates accuracy on test set
    // Returns: accuracy percentage (0-100)
}

int argmax(const vector<float>& vec) {
    // Finds index of maximum value
    // Used to get predicted digit from output probabilities
    // Input: probabilities (10)
    // Output: index of max (0-9)
}
```

### 🔨 Utility Functions (NEED TO BUILD)

```cpp
vector<float> initializeBias(int size) {
    // Creates bias vector initialized to small random values or zeros
    // Input: size
    // Output: bias vector
}

void saveWeights(const string& filename,
                 const vector<vector<float>>& W1,
                 const vector<float>& b1,
                 const vector<vector<float>>& W2,
                 const vector<float>& b2) {
    // Saves trained weights to file for later use
}
```

### 📊 Summary

**Completed:** 4 functions  
**To Build:** 15 functions  
**Total:** 19 functions

**Note:** Some functions from `matrixMath.h` and `machineLearning.h` are already available and can be reused!

---

# 🧱 **STEP 1 — Understanding MNIST Data**

### What is MNIST?

MNIST is a dataset of 70,000 handwritten digit images:
- 60,000 training images
- 10,000 test images
- Each image is 28×28 pixels (784 total)
- Each pixel is a grayscale value (0-255)

### The File Format

MNIST files are binary (not text). They have a specific structure:

**Images file structure:**
```
Bytes 0-3:   Magic number (2051) - file identifier
Bytes 4-7:   Number of images (60000)
Bytes 8-11:  Rows per image (28)
Bytes 12-15: Cols per image (28)
Bytes 16+:   Pixel data (one byte per pixel)
```

**Labels file structure:**
```
Bytes 0-3:   Magic number (2049)
Bytes 4-7:   Number of labels (60000)
Bytes 8+:    Labels (one byte per label, values 0-9)
```

### Why Binary Format?

- **Faster to load** than text
- **Smaller file size**
- **Standard format** for image datasets

### Reading Binary Data in C++

To read 4 bytes as an integer:

```cpp
int readInt(std::ifstream& file) {
    unsigned char bytes[4];
    file.read((char*)bytes, 4);
    // MNIST uses big-endian (most significant byte first)
    return (bytes[0] << 24) | (bytes[1] << 16) | 
           (bytes[2] << 8) | bytes[3];
}
```

**What this does:**
- Reads 4 bytes
- Combines them into a 32-bit integer
- Handles byte order (big-endian)

### Understanding How Bytes Are Stored

**Important:** These are **binary files**, not text files. When you open them in a text editor, they appear as gibberish - this is normal and expected!

#### What Does "Bytes 0-3" Mean?

Think of a binary file as a long sequence of bytes, numbered starting from 0:

```
Position:  0    1    2    3    4    5    6    7    8    9   10   11  ...
Value:    0x00 0x00 0x08 0x03 0x00 0x00 0xEA 0x60 0x00 0x00 0x00 0x1C ...
```

**"Bytes 0-3"** means: read 4 bytes starting at position 0 (positions 0, 1, 2, 3)

#### Images File - Detailed Byte Map

```
HEADER (16 bytes total):
├─ Bytes 0-3:   Magic number (0x00000803 = 2051 in decimal)
├─ Bytes 4-7:   Number of images (0x0000EA60 = 60000 in decimal)
├─ Bytes 8-11:  Rows (0x0000001C = 28 in decimal)
└─ Bytes 12-15: Columns (0x0000001C = 28 in decimal)

PIXEL DATA (starts at byte 16):
├─ Bytes 16-799:     First image (784 bytes = 28×28 pixels)
├─ Bytes 800-1583:   Second image (784 bytes)
├─ Bytes 1584-2367:  Third image (784 bytes)
└─ ... continues for all 60,000 images
```

**Total file size:** 16 + (60,000 × 784) = **47,040,016 bytes**

#### Labels File - Detailed Byte Map

```
HEADER (8 bytes total):
├─ Bytes 0-3:   Magic number (0x00000801 = 2049 in decimal)
└─ Bytes 4-7:   Number of labels (0x0000EA60 = 60000 in decimal)

LABEL DATA (starts at byte 8):
├─ Byte 8:      First label (0-9)
├─ Byte 9:      Second label (0-9)
├─ Byte 10:     Third label (0-9)
└─ ... continues for all 60,000 labels
```

**Total file size:** 8 + 60,000 = **60,008 bytes**

#### Why "Bytes 16+" and "Bytes 8+"?

The `+` means "and all bytes after this position until the end of file."

- **Images file:** After the 16-byte header, ALL remaining bytes are pixel data
- **Labels file:** After the 8-byte header, ALL remaining bytes are label data

#### Visualizing File Reading

When your C++ code reads the images file:

```cpp
ifstream file("train-images.idx3-ubyte", ios::binary);

// File pointer starts at byte 0
int magic = readInt(file);      // Reads bytes 0-3, pointer moves to byte 4
int numImages = readInt(file);  // Reads bytes 4-7, pointer moves to byte 8
int rows = readInt(file);       // Reads bytes 8-11, pointer moves to byte 12
int cols = readInt(file);       // Reads bytes 12-15, pointer moves to byte 16

// Now at byte 16 - start of pixel data
unsigned char pixels[784];
file.read((char*)pixels, 784);  // Reads bytes 16-799 (first image)
                                // Pointer now at byte 800

file.read((char*)pixels, 784);  // Reads bytes 800-1583 (second image)
                                // Pointer now at byte 1584
// ... and so on
```

The file pointer automatically advances as you read, so you don't need to manually calculate positions!

#### Common Confusion

❌ **Wrong thinking:** "Bytes 16+" means there's a byte with value 16  
✅ **Correct thinking:** "Bytes 16+" means starting at position 16 in the file

❌ **Wrong thinking:** The second file starts where the first file ends  
✅ **Correct thinking:** Images and labels are separate files, each starting at byte 0

#### Two Separate Files

Remember: **Images and labels are in different files!**

1. **`train-images.idx3-ubyte`** - Contains only image pixel data
2. **`train-labels.idx1-ubyte`** - Contains only labels

You need to read both files and match them up: `image[i]` corresponds to `label[i]`

#### Checking Your Files

You can verify your MNIST files are correct by checking their sizes:

```bash
ls -lh train-images.idx3-ubyte train-labels.idx1-ubyte t10k-images.idx3-ubyte t10k-labels.idx1-ubyte
```

**Expected sizes:**
- `train-images.idx3-ubyte`: 47,040,016 bytes (~45 MB)
- `train-labels.idx1-ubyte`: 60,008 bytes (~59 KB)
- `t10k-images.idx3-ubyte`: 7,840,016 bytes (~7.5 MB)
- `t10k-labels.idx1-ubyte`: 10,008 bytes (~10 KB)

### Exercise 1.1: Load One Image

**Goal:** Read the MNIST file and print the first image as ASCII art.

**Steps:**
1. Open file with `std::ifstream file(path, std::ios::binary)`
2. Read and verify magic number (should be 2051)
3. Read number of images, rows, cols
4. Read 784 bytes (28×28 pixels) for first image
5. Print each pixel as a character based on brightness

**Hints:**
- Use `file.read((char*)&variable, sizeof(variable))` to read binary data
- Normalize pixels: `float normalized = pixel / 255.0f`
- For ASCII art: if pixel > 128, print '█', else print ' '

**Expected output:** You should see a digit displayed in your terminal!

### Exercise 1.2: Load All Images

**Goal:** Load all 60,000 training images into memory.

**Data structure:**
```cpp
struct MNISTImage {
    std::vector<float> pixels;  // 784 values, normalized to [0, 1]
    int label;                   // 0-9
};
```

**Why this structure?**
- `vector<float>` for pixels: easy to manage, normalized for neural network
- `int` for label: simple and efficient

**Steps:**
1. Load all pixel data from images file
2. Load all labels from labels file
3. Combine them into `vector<MNISTImage>`
4. Verify: print `samples.size()` should be 60000

---

# 🧱 **STEP 2 — Representing Matrices**

### The Problem

Your OR neuron had 2 inputs. MNIST has 784 inputs!

You can't write:
```cpp
z = w1*x1 + w2*x2 + ... + w784*x784 + b  // Insane!
```

### The Solution: Matrix Operations

Store weights in a matrix, use matrix multiplication:

```
z = W × x + b
```

Where:
- `W` is a matrix of weights (784 × 128 for first layer)
- `x` is a vector of inputs (784 values)
- `z` is the output (128 values)

### Understanding Matrix-Vector Multiplication

**Example with small numbers:**

```
W = [1 2 3]    x = [1]
    [4 5 6]        [2]
                   [3]

W × x = [1*1 + 2*2 + 3*3]  = [14]
        [4*1 + 5*2 + 6*3]    [32]
```

Each output element is the dot product of a row with the input vector.

**In neural networks:**
- Each row of W represents one neuron's weights
- The dot product computes that neuron's weighted sum

### Exercise 2.1: Implement Matrix-Vector Multiply

**Goal:** Write a function that multiplies a matrix by a vector.

**Function signature:**
```cpp
void matVecMul(float* W, float* x, float* result, 
               int rows, int cols);
```

**What it should do:**
```
For each row i:
    result[i] = 0
    For each column j:
        result[i] += W[i * cols + j] * x[j]
```

**Remember:** Matrices are stored as flat arrays!
- `W[i * cols + j]` accesses element at row i, column j

**Test it:**
```cpp
float W[6] = {1, 2, 3, 4, 5, 6};  // 2×3 matrix
float x[3] = {1, 2, 3};
float result[2];

matVecMul(W, x, result, 2, 3);

// Should print: result[0] = 14, result[1] = 32
```

### Exercise 2.2: Add Bias

**Goal:** Add bias to each element of the result.

```cpp
void addBias(float* result, float* bias, int size) {
    // TODO: Add bias[i] to result[i] for each i
}
```

**Why separate function?**
- Clearer code
- Easier to debug
- Mirrors how neural networks are structured

---

# 🧱 **STEP 3 — Activation Functions**

### What You Already Know

From your OR neuron, you used sigmoid:
```cpp
float sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
}
```

### New Activation: ReLU

**Formula:** `ReLU(z) = max(0, z)`

**Why ReLU?**
- Simpler than sigmoid
- Faster to compute
- Works better for deep networks
- Avoids vanishing gradient problem

**Visual:**
```
Input:  [-2, -1, 0, 1, 2]
Output: [ 0,  0, 0, 1, 2]
```

### Exercise 3.1: Implement ReLU

**Goal:** Apply ReLU to an array of values.

```cpp
void relu(float* input, float* output, int size) {
    // TODO: For each element, output[i] = max(0, input[i])
}
```

**Test it:**
```cpp
float input[5] = {-2, -1, 0, 1, 2};
float output[5];
relu(input, output, 5);
// Should print: [0, 0, 0, 1, 2]
```

### New Activation: Softmax

**What it does:** Converts scores to probabilities that sum to 1.

**Example:**
```
Input:  [1, 2, 3]
Output: [0.09, 0.24, 0.67]  // Sum = 1.0
```

**Formula:**
```
softmax(z)[i] = exp(z[i]) / sum(exp(z[j]) for all j)
```

**Why softmax?**
- Output layer needs probabilities
- Each output represents "confidence this is digit X"
- Probabilities must sum to 1

### Exercise 3.2: Implement Softmax

**Goal:** Convert scores to probabilities.

**Steps:**
1. Find max value (for numerical stability)
2. Compute exp(z[i] - max) for each element
3. Sum all exponentials
4. Divide each by the sum

**Numerical stability trick:**
- `exp(z)` can overflow for large z
- `exp(z - max)` keeps values reasonable
- Mathematically equivalent!

---

# 🧱 **STEP 4 — Forward Pass**

### The Concept

Forward pass = computing the prediction.

**For one layer:**
```
1. Compute weighted sum: z = W × x + b
2. Apply activation: a = activation(z)
```

**For your network:**
```
Input (784) 
    → Layer 1: z1 = W1 × input + b1
    → ReLU: h = ReLU(z1)  [128 values]
    → Layer 2: z2 = W2 × h + b2
    → Softmax: output = softmax(z2)  [10 values]
```

### Exercise 4.1: Implement One Layer Forward

**Goal:** Compute output of one dense layer.

```cpp
void denseLayerForward(
    float* input, int inputSize,
    float* weights, float* bias, int outputSize,
    float* output,
    void (*activation)(float*, float*, int)
) {
    // TODO:
    // 1. Matrix-vector multiply: temp = weights × input
    // 2. Add bias
    // 3. Apply activation function
}
```

**Why pass activation as a function pointer?**
- Reusable for different activations
- Same code works for ReLU, sigmoid, softmax

### Exercise 4.2: Full Network Forward Pass

**Goal:** Compute prediction for one image.

**Pseudocode:**
```
Function predict(image):
    h = denseLayerForward(image, W1, b1, relu)
    output = denseLayerForward(h, W2, b2, softmax)
    return output
```

**Test it:**
- Initialize weights randomly
- Pass in one MNIST image
- Should get 10 probabilities that sum to 1

---

# 🧱 **STEP 5 — Computing Loss**

### What You Already Know

From your OR neuron, you used binary cross-entropy:
```cpp
loss = -(y * log(y_hat) + (1-y) * log(1-y_hat))
```

### Categorical Cross-Entropy

For 10 classes (digits 0-9), we use categorical cross-entropy:

```
loss = -sum(y[i] * log(y_hat[i]) for i in 0..9)
```

**Example:**
```
True label: 3 → y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (one-hot)
Prediction:     ŷ = [0.1, 0.05, 0.05, 0.7, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005]

loss = -(0*log(0.1) + 0*log(0.05) + ... + 1*log(0.7) + ... + 0*log(0.005))
     = -log(0.7)
     = 0.357
```

**Interpretation:**
- Only the true class contributes to loss
- Predicted 0.7 for digit 3 → loss = 0.357 (okay)
- If predicted 0.9 → loss = 0.105 (better!)
- If predicted 0.1 → loss = 2.303 (bad!)

### Exercise 5.1: Implement Cross-Entropy Loss

**Goal:** Compute loss for one prediction.

```cpp
float crossEntropyLoss(float* prediction, int trueLabel) {
    // TODO:
    // Return -log(prediction[trueLabel])
    // Add small epsilon to avoid log(0)
}
```

**Test it:**
```cpp
float pred[10] = {0.1, 0.05, 0.05, 0.7, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005};
float loss = crossEntropyLoss(pred, 3);
// Should print: ~0.357
```

---

# 🧱 **STEP 6 — Backward Pass (Backpropagation)**

### The Concept

You already understand this from Phase 1! The chain rule.

**For your OR neuron:**
```
error = y_hat - y
dL/dw = error * x
```

**For a multi-layer network:**
Same idea, but we propagate errors backward through layers.

### Layer 2 (Output Layer)

**Gradient w.r.t. output:**
```
dL/dz2 = output - y_true  (for softmax + cross-entropy)
```

**This is the same as your OR neuron!** Just with 10 outputs instead of 1.

**Gradient w.r.t. weights:**
```
dL/dW2 = h^T × dL/dz2
```

Where `h` is the hidden layer output (from forward pass).

### Layer 1 (Hidden Layer)

**Gradient propagates backward:**
```
dL/dh = dL/dz2 × W2^T
```

**Through ReLU:**
```
dL/dz1 = dL/dh ⊙ ReLU'(z1)
```

Where `⊙` means element-wise multiply, and `ReLU'(z) = 1 if z > 0, else 0`.

**Gradient w.r.t. weights:**
```
dL/dW1 = input^T × dL/dz1
```

### Exercise 6.1: Implement ReLU Backward

**Goal:** Compute gradient through ReLU.

```cpp
void reluBackward(float* z, float* gradOutput, float* gradInput, int size) {
    // TODO:
    // gradInput[i] = gradOutput[i] if z[i] > 0, else 0
}
```

**Why we need z:**
- ReLU derivative depends on the input
- If input was negative, gradient is 0
- If input was positive, gradient passes through

### Exercise 6.2: Implement Layer Backward

**Goal:** Compute gradients for one layer and update weights.

**Inputs:**
- `gradOutput`: gradient from next layer
- `input`: input to this layer (saved from forward pass)
- `weights`, `bias`: layer parameters
- `learningRate`: how much to update

**Outputs:**
- `gradInput`: gradient to pass to previous layer
- Updated `weights` and `bias`

**Steps:**
1. Compute gradient through activation
2. Compute gradient w.r.t. weights: `input^T × gradActivation`
3. Compute gradient w.r.t. input: `gradActivation × weights^T`
4. Update weights: `weights -= learningRate * gradWeights`
5. Update bias: `bias -= learningRate * gradBias`

---

# 🧱 **STEP 7 — Training Loop**

### The Structure (Same as OR Neuron!)

```cpp
for each epoch:
    for each training sample:
        1. Forward pass: compute prediction
        2. Compute loss
        3. Backward pass: compute gradients
        4. Update weights
    
    Print average loss
```

### Mini-Batch Training

**Problem:** 60,000 samples is a lot!

**Solution:** Process in batches of 32.

**Why batches?**
- Faster (can parallelize)
- More stable gradients (average over batch)
- Standard practice in deep learning

### Exercise 7.1: Implement Training Loop

**Goal:** Train your network for 10 epochs.

**Pseudocode:**
```
Initialize weights randomly

for epoch in 1..10:
    totalLoss = 0
    
    for batch in training data (batch size 32):
        # Forward pass for entire batch
        predictions = forward(batch)
        
        # Compute loss
        loss = crossEntropyLoss(predictions, labels)
        totalLoss += loss
        
        # Backward pass
        backward(predictions, labels)
        
        # Update weights (done in backward)
    
    print "Epoch", epoch, "Loss:", totalLoss / numBatches
```

**What to expect:**
- Epoch 1: Loss ~2.3 (random guessing)
- Epoch 5: Loss ~0.5 (learning!)
- Epoch 10: Loss ~0.3 (good!)

---

# 🧱 **STEP 8 — Testing**

### The Process

1. Load test data (10,000 images)
2. For each image:
   - Forward pass (no training!)
   - Find predicted digit (argmax of output)
   - Compare with true label
3. Compute accuracy

### Exercise 8.1: Implement Testing

**Goal:** Measure accuracy on test set.

```cpp
float test(testData, W1, b1, W2, b2) {
    int correct = 0
    
    for each sample in testData:
        output = forward(sample.pixels)
        
        # Find predicted digit (index of max probability)
        predicted = argmax(output)
        
        if predicted == sample.label:
            correct++
    
    return (float)correct / testData.size() * 100
}
```

**Expected accuracy:** 90-95% after 10 epochs!

---

# 🧱 **STEP 9 — Putting It All Together**

### Main Program Structure

```cpp
int main() {
    // 1. Load data
    auto trainData = loadMNIST("train-images", "train-labels");
    auto testData = loadMNIST("test-images", "test-labels");
    
    // 2. Initialize network
    float* W1 = randomWeights(784, 128);
    float* b1 = zeros(128);
    float* W2 = randomWeights(128, 10);
    float* b2 = zeros(10);
    
    // 3. Train
    train(trainData, W1, b1, W2, b2, epochs=10, lr=0.01);
    
    // 4. Test
    float accuracy = test(testData, W1, b1, W2, b2);
    cout << "Accuracy: " << accuracy << "%" << endl;
    
    return 0;
}
```

### Weight Initialization

**Important:** Don't initialize to zero!

**Why?**
- All neurons would compute the same thing
- No learning would happen

**Solution:** Random small values.

```cpp
float* randomWeights(int rows, int cols) {
    float* W = new float[rows * cols];
    for (int i = 0; i < rows * cols; i++) {
        // Xavier initialization
        W[i] = (rand() / (float)RAND_MAX - 0.5) * sqrt(6.0 / (rows + cols));
    }
    return W;
}
```

---

## ✅ **Phase 2 Checkpoint**

Before moving to Phase 3, you should be able to:

- [ ] Load MNIST images and labels from binary files
- [ ] Implement matrix-vector multiplication
- [ ] Implement ReLU and Softmax activations
- [ ] Compute forward pass through 2-layer network
- [ ] Compute cross-entropy loss
- [ ] Implement backpropagation for both layers
- [ ] Train the network and see loss decrease
- [ ] Achieve 90%+ accuracy on test set

### Expected Results

After 10 epochs:
- Training loss: ~0.3
- Test accuracy: 90-95%
- Training time: 5-10 minutes on CPU

---

## 🎯 **Next Steps**

1. ✅ Experiment with hyperparameters (learning rate, hidden size)
2. ✅ Try adding more layers
3. ✅ Save and load trained weights
4. ✅ Move to **Phase 3: CUDA Fundamentals**

**Congratulations!** You've built a neural network from scratch! 🎉

Now you're ready to make it run on the GPU!
