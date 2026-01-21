# 🚀 **Phase 4: GPU Neural Network**

## Philosophy

**You cannot optimize what you don't understand.**

This phase combines everything: neural network knowledge from Phase 2 and CUDA skills from Phase 3. You'll port your CPU neural network to run on the GPU.

> **Prerequisites:** This project assumes you've completed **Phase 2: CPU Neural Network** (for neural network understanding) and **Phase 3: CUDA Fundamentals** (for GPU programming skills). However, Phase 4 is a standalone project with its own implementation and can be understood independently if you have the prerequisite knowledge.
>
> **Related Projects:**
> - **Phase 2:** `../Phase-2-CPU-Neural-Net/` - CPU implementation of the neural network
> - **Phase 3:** `../Phase-3-CUDA-Fundamentals/` - CUDA fundamentals and GPU programming basics

---

## 🎯 **Goal**

Accelerate your MNIST neural network using CUDA. Achieve the same 93-95% accuracy, but **10-50x faster!**

**Time:** 5-7 days  
**Difficulty:** ⭐⭐⭐⭐⭐

---

## 🤔 **Why This Phase Matters**

- This is the **culmination** of everything you've learned
- You'll see **dramatic speedups** (minutes → seconds)
- You'll understand how PyTorch/TensorFlow work under the hood
- You'll have built a complete ML system from scratch

---

## 📚 **What You'll Port to GPU**

From Phase 2 (CPU Neural Network), these operations will be accelerated on GPU using the CUDA concepts from Phase 3:

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Matrix multiply (W1 × input) | ~2ms | ~0.1ms | 20x |
| ReLU (128 elements) | ~0.01ms | ~0.001ms | 10x |
| Matrix multiply (W2 × hidden) | ~0.1ms | ~0.01ms | 10x |
| Softmax (10 elements) | ~0.001ms | ~0.001ms | 1x |
| Backprop | ~3ms | ~0.2ms | 15x |
| **Total per image** | **~5ms** | **~0.3ms** | **~17x** |

**For 60,000 images:** 5 minutes → 20 seconds! 🚀

---

# 🧱 **STEP 1 — Strategy: What Stays, What Moves**

> **Concepts from Phase 3:** This step applies the memory management patterns you learned in Phase 3 (host vs device memory, `cudaMalloc`, `cudaMemcpy`). If you need a refresher, see `../Phase-3-CUDA-Fundamentals/README.md` Step 5.

### Keep on CPU (Host)

```cpp
// Control flow
for (int epoch = 0; epoch < 10; epoch++) {
    for (int batch = 0; batch < numBatches; batch++) {
        // Launch GPU kernels here
    }
    printf("Epoch %d complete\n", epoch);  // Printing
}

// Data loading
loadMNISTImages(...);  // File I/O stays on CPU
```

### Move to GPU (Device)

```cpp
// All math operations
- Matrix multiplication (W × input)
- Bias addition (z + b)
- ReLU activation
- Softmax activation
- Loss computation
- Gradient computation
- Weight updates
```

### Memory Strategy

**Allocate once, keep on GPU:**

```cpp
// At start of training
float *d_W1, *d_b1, *d_W2, *d_b2;  // Weights stay on GPU
cudaMalloc(&d_W1, 128 * 784 * sizeof(float));
cudaMalloc(&d_b1, 128 * sizeof(float));
cudaMalloc(&d_W2, 10 * 128 * sizeof(float));
cudaMalloc(&d_b2, 10 * sizeof(float));

// Copy initial weights once
cudaMemcpy(d_W1, h_W1, ..., cudaMemcpyHostToDevice);
// ... train for 10 epochs, weights stay on GPU ...

// At end, copy back once
cudaMemcpy(h_W1, d_W1, ..., cudaMemcpyDeviceToHost);
```

**Why?** Copying CPU ↔ GPU is **slow**. Keep data on GPU as long as possible!

---

# 🧱 **STEP 2 — GPU Forward Pass**

> **Building on Phase 3:** This step uses the matrix-vector multiplication kernel pattern from Phase 3 Step 7. You'll extend those concepts to handle batched operations.

### Phase 2 Forward Pass (CPU)

```cpp
// Layer 1
vector<float> hiddenZ = processLayer(w1, b1, images[i]);  // W1 × input + b1
vector<float> hiddenA = relu(hiddenZ);                    // ReLU

// Layer 2
vector<float> outputZ = processLayer(w2, b2, hiddenA);    // W2 × hidden + b2
vector<float> y_hat = softmax(outputZ);                   // Softmax
```

### Phase 4 Forward Pass (GPU)

```cpp
// All data already on GPU: d_input, d_W1, d_b1, d_W2, d_b2

// Layer 1: W1 × input + b1 → hiddenZ
matVecMulKernel<<<1, 128>>>(d_W1, d_input, d_hiddenZ, 128, 784);
addBiasKernel<<<1, 128>>>(d_hiddenZ, d_b1, 128);
reluKernel<<<1, 128>>>(d_hiddenZ, d_hiddenA, 128);

// Layer 2: W2 × hidden + b2 → outputZ
matVecMulKernel<<<1, 10>>>(d_W2, d_hiddenA, d_outputZ, 10, 128);
addBiasKernel<<<1, 10>>>(d_outputZ, d_b2, 10);
softmaxKernel<<<1, 1>>>(d_outputZ, d_y_hat, 10);
```

**Same operations, different location!**

### New Kernel: Add Bias

```cpp
__global__ void addBiasKernel(float* z, float* bias, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        z[idx] += bias[idx];  // In-place addition
    }
}
```

### New Kernel: Softmax

```cpp
__global__ void softmaxKernel(float* input, float* output, int n) {
    // Note: This is a simplified version for small n (like 10)
    // For one thread to handle entire softmax
    
    if (threadIdx.x == 0) {  // Only thread 0 does the work
        // Find max for numerical stability
        float maxVal = input[0];
        for (int i = 1; i < n; i++) {
            if (input[i] > maxVal) maxVal = input[i];
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            output[i] = expf(input[i] - maxVal);
            sum += output[i];
        }
        
        // Normalize
        for (int i = 0; i < n; i++) {
            output[i] /= sum;
        }
    }
}
```

**Why one thread?** Softmax requires global sum - easier with one thread for small arrays.

### Exercise 2.1: Implement Forward Pass

**Goal:** Run forward pass entirely on GPU for one image.

**Steps:**
1. Copy one image from CPU to GPU
2. Launch forward pass kernels
3. Copy output back to CPU
4. Compare with Phase 2 CPU version
5. Outputs should match!

---

# 🧱 **STEP 3 — GPU Backward Pass**

### Phase 2 Backward Pass (CPU)

```cpp
// Output layer error
vector<float> outputError(10);
for (int i = 0; i < 10; i++) {
    outputError[i] = y_hat[i] - (i == y ? 1.0f : 0.0f);
}

// Update W2
for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 128; j++) {
        w2[i][j] -= learning_rate * (outputError[i] * hiddenA[j]);
    }
}

// Backprop to hidden
vector<float> hiddenError(128, 0.0f);
for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 10; j++) {
        hiddenError[i] += outputError[j] * w2[j][i];
    }
}

// Update W1
for (int i = 0; i < 128; i++) {
    float relu_deriv = (hiddenZ[i] > 0) ? 1.0f : 0.0f;
    for (int j = 0; j < 784; j++) {
        w1[i][j] -= learning_rate * (hiddenError[i] * relu_deriv * images[idx][j]);
    }
}
```

### Phase 4 Backward Pass (GPU)

**Step 1: Compute output error**
```cpp
__global__ void computeOutputErrorKernel(float* y_hat, int trueLabel, 
                                          float* error, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        error[idx] = y_hat[idx] - (idx == trueLabel ? 1.0f : 0.0f);
    }
}
```

**Step 2: Update W2 (outer product)**
```cpp
__global__ void updateW2Kernel(float* W2, float* outputError, float* hiddenA,
                                float lr, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // W2[row][col] -= lr * outputError[row] * hiddenA[col]
        W2[row * cols + col] -= lr * outputError[row] * hiddenA[col];
    }
}

// Launch with 2D grid
dim3 threads(16, 16);
dim3 blocks((128 + 15) / 16, (10 + 15) / 16);
updateW2Kernel<<<blocks, threads>>>(d_W2, d_outputError, d_hiddenA, lr, 10, 128);
```

**Step 3: Backprop to hidden (W2ᵀ × error)**
```cpp
__global__ void backpropToHiddenKernel(float* W2, float* outputError,
                                        float* hiddenError, int hiddenSize, int outputSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < hiddenSize) {
        float sum = 0.0f;
        for (int j = 0; j < outputSize; j++) {
            sum += outputError[j] * W2[j * hiddenSize + idx];  // W2 transposed
        }
        hiddenError[idx] = sum;
    }
}
```

**Step 4: Apply ReLU derivative**
```cpp
__global__ void reluBackwardKernel(float* hiddenZ, float* hiddenError, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        if (hiddenZ[idx] <= 0) {
            hiddenError[idx] = 0.0f;  // ReLU derivative is 0 for negative inputs
        }
        // Otherwise, gradient passes through unchanged
    }
}
```

**Step 5: Update W1**
```cpp
__global__ void updateW1Kernel(float* W1, float* hiddenError, float* input,
                                float lr, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        W1[row * cols + col] -= lr * hiddenError[row] * input[col];
    }
}

// Launch with 2D grid for large matrix
dim3 threads(16, 16);
dim3 blocks((784 + 15) / 16, (128 + 15) / 16);
updateW1Kernel<<<blocks, threads>>>(d_W1, d_hiddenError, d_input, lr, 128, 784);
```

### Complete Backward Pass

```cpp
void backwardPassGPU(float* d_y_hat, int trueLabel, float* d_W2, float* d_b2,
                     float* d_hiddenA, float* d_hiddenZ, float* d_W1, float* d_b1,
                     float* d_input, float lr) {
    
    // Allocate temporary arrays for errors
    float *d_outputError, *d_hiddenError;
    cudaMalloc(&d_outputError, 10 * sizeof(float));
    cudaMalloc(&d_hiddenError, 128 * sizeof(float));
    
    // 1. Compute output error
    computeOutputErrorKernel<<<1, 10>>>(d_y_hat, trueLabel, d_outputError, 10);
    
    // 2. Update W2 and b2
    dim3 threads2(16, 16);
    dim3 blocks2((128 + 15) / 16, (10 + 15) / 16);
    updateW2Kernel<<<blocks2, threads2>>>(d_W2, d_outputError, d_hiddenA, lr, 10, 128);
    updateBiasKernel<<<1, 10>>>(d_b2, d_outputError, lr, 10);
    
    // 3. Backprop to hidden
    backpropToHiddenKernel<<<1, 128>>>(d_W2, d_outputError, d_hiddenError, 128, 10);
    
    // 4. Apply ReLU derivative
    reluBackwardKernel<<<1, 128>>>(d_hiddenZ, d_hiddenError, 128);
    
    // 5. Update W1 and b1
    dim3 threads1(16, 16);
    dim3 blocks1((784 + 15) / 16, (128 + 15) / 16);
    updateW1Kernel<<<blocks1, threads1>>>(d_W1, d_hiddenError, d_input, lr, 128, 784);
    updateBiasKernel<<<1, 128>>>(d_b1, d_hiddenError, lr, 128);
    
    // Free temporary memory
    cudaFree(d_outputError);
    cudaFree(d_hiddenError);
}
```

### Update Bias Kernel

```cpp
__global__ void updateBiasKernel(float* bias, float* error, float lr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        bias[idx] -= lr * error[idx];
    }
}
```

---

# 🧱 **STEP 4 — Batch Processing**

### The Problem

Processing one image at a time wastes GPU power!

**Current:**
```
GPU: [████░░░░░░░░░░░░░░░░] 20% utilized
```

**With batches:**
```
GPU: [████████████████████] 100% utilized
```

### Batch Forward Pass

**Instead of:**
```
Input: 1 × 784 (one image)
W1: 128 × 784
Output: 1 × 128
```

**Use:**
```
Input: 32 × 784 (batch of 32 images)
W1: 128 × 784
Output: 32 × 128 (32 results)
```

### Batch Matrix Multiply Kernel

```cpp
__global__ void batchMatMulKernel(float* input, float* weights, float* output,
                                   int batchSize, int inputSize, int outputSize) {
    int batch = blockIdx.y;  // Which image in batch
    int outIdx = threadIdx.x + blockIdx.x * blockDim.x;  // Which output neuron
    
    if (batch < batchSize && outIdx < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            sum += input[batch * inputSize + i] * weights[outIdx * inputSize + i];
        }
        output[batch * outputSize + outIdx] = sum;
    }
}

// Launch
dim3 threads(256, 1);
dim3 blocks((128 + 255) / 256, 32);  // 32 batches
batchMatMulKernel<<<blocks, threads>>>(d_input, d_W1, d_hidden, 32, 784, 128);
```

### Exercise 4.1: Implement Batch Processing

**Goal:** Process 32 images simultaneously.

**Steps:**
1. Allocate memory for batch: `32 × 784` floats
2. Copy 32 images to GPU
3. Launch batch kernels
4. Process all 32 in one go!

**Expected speedup:** 2-5x over single-image processing

---

# 🧱 **STEP 5 — Complete Training Loop**

### Structure

```cpp
int main() {
    // 1. Load data on CPU
    vector<vector<float>> trainImages = loadMNISTImages(...);
    vector<int> trainLabels = loadMNISTLabels(...);
    
    // 2. Allocate GPU memory (ONCE)
    float *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_input, *d_hiddenZ, *d_hiddenA, *d_outputZ, *d_y_hat;
    
    cudaMalloc(&d_W1, 128 * 784 * sizeof(float));
    cudaMalloc(&d_b1, 128 * sizeof(float));
    cudaMalloc(&d_W2, 10 * 128 * sizeof(float));
    cudaMalloc(&d_b2, 10 * sizeof(float));
    cudaMalloc(&d_input, 784 * sizeof(float));
    cudaMalloc(&d_hiddenZ, 128 * sizeof(float));
    cudaMalloc(&d_hiddenA, 128 * sizeof(float));
    cudaMalloc(&d_outputZ, 10 * sizeof(float));
    cudaMalloc(&d_y_hat, 10 * sizeof(float));
    
    // 3. Initialize and copy weights to GPU
    initializeWeights(h_W1, h_b1, h_W2, h_b2);
    cudaMemcpy(d_W1, h_W1, 128 * 784 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, 10 * 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, 10 * sizeof(float), cudaMemcpyHostToDevice);
    
    // 4. Training loop
    float learningRate = 0.1f;
    for (int epoch = 0; epoch < 10; epoch++) {
        float totalLoss = 0.0f;
        
        for (int i = 0; i < trainImages.size(); i++) {
            // Copy image to GPU
            cudaMemcpy(d_input, trainImages[i].data(), 784 * sizeof(float), 
                      cudaMemcpyHostToDevice);
            
            // Forward pass (all on GPU)
            forwardPassGPU(d_input, d_W1, d_b1, d_W2, d_b2,
                          d_hiddenZ, d_hiddenA, d_outputZ, d_y_hat);
            
            // Compute loss (copy output back temporarily)
            float h_y_hat[10];
            cudaMemcpy(h_y_hat, d_y_hat, 10 * sizeof(float), cudaMemcpyDeviceToHost);
            totalLoss += categoricalCrossEntropy(h_y_hat, trainLabels[i]);
            
            // Backward pass (all on GPU)
            backwardPassGPU(d_y_hat, trainLabels[i], d_W2, d_b2,
                           d_hiddenA, d_hiddenZ, d_W1, d_b1,
                           d_input, learningRate);
        }
        
        printf("Epoch %d/%d - Loss: %.4f\n", epoch + 1, 10, totalLoss / trainImages.size());
    }
    
    // 5. Copy trained weights back to CPU
    cudaMemcpy(h_W1, d_W1, 128 * 784 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b1, d_b1, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W2, d_W2, 10 * 128 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2, d_b2, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 6. Save weights
    saveWeights("gpu_weights", h_W1, h_b1, h_W2, h_b2);
    
    // 7. Free GPU memory
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_input); cudaFree(d_hiddenZ); cudaFree(d_hiddenA);
    cudaFree(d_outputZ); cudaFree(d_y_hat);
    
    return 0;
}
```

---

# 🧱 **STEP 6 — Benchmarking**

### Measuring GPU Performance

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... your GPU code ...
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("GPU time: %.2f ms\n", milliseconds);
```

### What to Measure

```cpp
// Measure forward pass
cudaEventRecord(start);
forwardPassGPU(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&forwardTime, start, stop);

// Measure backward pass
cudaEventRecord(start);
backwardPassGPU(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&backwardTime, start, stop);

// Measure total epoch
auto cpuStart = std::chrono::high_resolution_clock::now();
// ... train one epoch ...
auto cpuEnd = std::chrono::high_resolution_clock::now();
auto epochTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart);
```

### Expected Results

| Operation | CPU (Phase 2) | GPU (Phase 4) | Speedup |
|-----------|---------------|---------------|---------|
| Forward pass (1 image) | 2ms | 0.1ms | 20x |
| Backward pass (1 image) | 3ms | 0.2ms | 15x |
| One epoch (60k images) | 300s | 20s | 15x |
| Total training (10 epochs) | 50min | 3min | 17x |

**Your results may vary based on GPU!**

---

## ✅ **Phase 4 Checkpoint**

**Current Implementation Status:**

This project implements the **complete GPU-accelerated neural network**:

- ✅ **Full GPU Forward Pass**: `batchedForwardPass` processes entire batches on GPU
  - Batched matrix-vector multiplication
  - Batched ReLU activation
  - Batched bias addition
  - All forward operations on GPU

- ✅ **Full GPU Backward Pass**: `batchedBackwardPass` computes gradients and updates weights entirely on GPU
  - GPU softmax activation
  - GPU output error computation
  - GPU gradient computation (W1, b1, W2, b2)
  - GPU backpropagation to hidden layer
  - GPU ReLU backward (derivative application)
  - GPU weight and bias updates
  - **No CPU↔GPU transfers during training** (except for loss computation)

- ✅ **Complete Training Loop**: Entire training process runs on GPU
  - Weights stay on GPU throughout training
  - Batched processing (32 samples at once)
  - Minimal host-device transfers

- ✅ **Performance**: Achieves 10-50x speedup over CPU implementation

**What's Implemented:**
- [x] GPU forward pass working correctly (batched)
- [x] GPU backward pass working correctly (batched)
- [x] Training loop running entirely on GPU
- [x] GPU softmax activation
- [x] GPU gradient computation
- [x] GPU weight updates
- [x] Batched processing for maximum performance
- [x] Weight management on GPU (minimal transfers)

**Comparison with Phase 3:**
- **Phase 3**: Forward pass on GPU, backward pass on CPU (hybrid)
- **Phase 4**: Forward AND backward pass entirely on GPU (complete)

### Verification

**Run the GPU version:**
```bash
# Phase 4 (GPU - Complete Implementation)
cd Phase-4-GPU-Neural-Net
./startTraining.sh
# Or directly: ./build/gpu_neuron
```

**Expected Results:**
- Training time: 10-50x faster than CPU (Phase 2)
- Accuracy: 93-95% (same as Phase 2)
- All operations run on GPU (forward + backward)
- Minimal CPU↔GPU transfers (only for loss computation)

**Compare with Phase 3:**
```bash
# Phase 3 (Hybrid: GPU forward, CPU backward)
cd Phase-3-CUDA-Fundamentals
./startTraining.sh
# Note: Uses GPU for forward pass only

# Phase 4 (Complete: GPU forward + backward)
cd Phase-4-GPU-Neural-Net
./startTraining.sh
# Complete GPU implementation
```

**Key Differences:**
- **Phase 3**: Hybrid approach (forward on GPU, backward on CPU) - good for learning
- **Phase 4**: Complete GPU implementation (forward + backward on GPU) - production-ready
- **Phase 4** should be faster than Phase 3 due to GPU backward pass

---

## 🎯 **Next Steps**

**You've accomplished something incredible:**
- ✅ Built a neural network from scratch (Phase 2)
- ✅ Learned GPU programming fundamentals (Phase 3)
- ✅ Built complete GPU-accelerated neural network (Phase 4) ← **You are here!**

**What You've Built:**
- Complete forward pass on GPU (batched)
- Complete backward pass on GPU (batched)
- GPU weight updates (no CPU transfers)
- 10-50x speedup over CPU
- Production-ready implementation

**Future Enhancements (Optional):**
- Optimize further (shared memory, better kernels)
- Add features (Adam optimizer, batch normalization)
- Build cool demos (real-time digit recognition)
- Experiment with architectures (deeper networks, CNNs)
- Multi-GPU training
- Mixed precision training (FP16)

**Or you're done!** You've built a complete, working, GPU-accelerated neural network from scratch. That's a massive achievement! 🎉

**Project Status:**
- **Phase 2**: ✅ Complete CPU implementation
- **Phase 3**: ✅ GPU forward pass + CPU backward pass (hybrid)
- **Phase 4**: ✅ Complete GPU implementation (forward + backward) ← **Current**

---

## 💡 **Key Takeaways**

**The Learning Path:**
```
Phase 2: Understand the math (CPU Neural Network)
Phase 3: Learn GPU programming (CUDA Fundamentals)  
Phase 4: Combine them (GPU Neural Network) ← You are here!
```

**This pattern applies to ANY ML algorithm:**
1. Implement on CPU first (Phase 2)
2. Learn GPU programming fundamentals (Phase 3)
3. Port to GPU with full optimization (Phase 4)
4. Profit! 🚀

**Project Relationships:**
- **Phase 2** (`../Phase-2-CPU-Neural-Net/`) provides the neural network foundation
- **Phase 3** (`../Phase-3-CUDA-Fundamentals/`) provides the GPU programming skills
- **Phase 4** (this project) combines both into a production-ready GPU-accelerated system

**You now understand how PyTorch/TensorFlow work under the hood!**

**Congratulations!** You've built a GPU-accelerated neural network from scratch! 🚀
