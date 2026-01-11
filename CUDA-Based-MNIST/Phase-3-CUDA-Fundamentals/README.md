# ⚡ **Phase 3: CUDA Fundamentals**

## Philosophy

**You cannot parallelize what you don't understand.**

This phase teaches you GPU programming from first principles. You'll learn how to think in parallel and write kernels that run on thousands of cores simultaneously.

---

## 🎯 **Goal**

Master CUDA basics by writing simple parallel programs. Understand threads, blocks, memory, and how to accelerate math operations on the GPU.

**Time:** 5-7 days  
**Difficulty:** ⭐⭐⭐⭐

---

## 🤔 **Why This Phase Matters**

- Your M1 has **thousands of GPU cores** (vs 8 CPU cores)
- Neural networks are **embarrassingly parallel** - same operation on millions of numbers
- Matrix multiplication: **perfect for GPU!**
- Understanding CUDA fundamentals makes Phase 4 trivial

---

## 📚 **What You'll Learn**

1. How GPUs differ from CPUs
2. Thread indexing (the key to everything!)
3. Memory management (host ↔ device)
4. Writing CUDA kernels
5. Matrix operations on GPU
6. Debugging GPU code

**Same concepts as Phase 2, just running on 1000+ cores at once!**

---

# 🧱 **STEP 1 — CPU vs GPU: The Mental Model**

### Your M1 MacBook

**CPU (8 cores):**
```
Core 1: [████████] Very fast, complex tasks
Core 2: [████████] Can do anything
Core 3: [████████] Good at branching
...
Core 8: [████████] 
```

**GPU (1000+ cores):**
```
Core 1:   [█] Simple, specialized
Core 2:   [█] Same operation
Core 3:   [█] No complex branching
...
Core 1000:[█] But MANY of them!
```

### The Trade-off

| Feature | CPU | GPU |
|---------|-----|-----|
| **Cores** | 8 | 1000+ |
| **Speed per core** | Very fast | Slower |
| **Best for** | Complex logic | Simple, repeated operations |
| **Your Phase 2 code** | ✅ Works great | ❌ Not optimized |

### When GPU Wins

**Your Phase 2 matrix multiply:**
```cpp
// CPU: One core does ALL 100,352 multiplications
for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 784; j++) {
        result[i] += w1[i][j] * input[j];  // Sequential
    }
}
```

**GPU version:**
```cpp
// GPU: 128 cores each do 784 multiplications SIMULTANEOUSLY
// Core 0 handles row 0, Core 1 handles row 1, etc.
```

**Result:** 10-50x faster!

---

# 🧱 **STEP 2 — Your First CUDA Program**

### The Simplest Kernel

Create `hello.cu`:

```cpp
#include <stdio.h>

__global__ void helloKernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    helloKernel<<<1, 10>>>();  // Launch 10 threads
    cudaDeviceSynchronize();    // Wait for GPU to finish
    return 0;
}
```

### Compile and Run

```bash
nvcc hello.cu -o hello
./hello
```

**Expected output:**
```
Hello from GPU thread 0!
Hello from GPU thread 1!
Hello from GPU thread 2!
...
Hello from GPU thread 9!
```

(Order may vary - threads run in parallel!)

### Breaking It Down

**`__global__`** = "This function runs on GPU"
- Called from CPU
- Executed on GPU
- Many copies run in parallel

**`<<<1, 10>>>`** = Launch configuration
- `1` = number of blocks
- `10` = threads per block
- Total threads = 1 × 10 = 10

**`threadIdx.x`** = Thread's ID within its block (0-9)

**`cudaDeviceSynchronize()`** = Wait for GPU to finish
- GPU runs asynchronously
- Without this, program might exit before GPU finishes

---

# 🧱 **STEP 3 — Thread Indexing: The Key to Everything**

### The Problem You're Solving

You have 60,000 images to process. How does each GPU thread know which image to work on?

**Answer:** Thread indexing!

### The Formula (Memorize This!)

```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

**What each part means:**
- `threadIdx.x` = Thread number within block (0, 1, 2, ...)
- `blockIdx.x` = Block number (0, 1, 2, ...)
- `blockDim.x` = Threads per block (constant you choose)

### Visual Example

Launch: `myKernel<<<3, 4>>>()`  (3 blocks, 4 threads each)

```
Block 0:              Block 1:              Block 2:
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Thread 0    │      │ Thread 0    │      │ Thread 0    │
│ idx = 0     │      │ idx = 4     │      │ idx = 8     │
├─────────────┤      ├─────────────┤      ├─────────────┤
│ Thread 1    │      │ Thread 1    │      │ Thread 1    │
│ idx = 1     │      │ idx = 5     │      │ idx = 9     │
├─────────────┤      ├─────────────┤      ├─────────────┤
│ Thread 2    │      │ Thread 2    │      │ Thread 2    │
│ idx = 2     │      │ idx = 6     │      │ idx = 10    │
├─────────────┤      ├─────────────┤      ├─────────────┤
│ Thread 3    │      │ Thread 3    │      │ Thread 3    │
│ idx = 3     │      │ idx = 7     │      │ idx = 11    │
└─────────────┘      └─────────────┘      └─────────────┘
```

**Calculation examples:**
- Block 0, Thread 0: `idx = 0 + 0*4 = 0`
- Block 0, Thread 3: `idx = 3 + 0*4 = 3`
- Block 1, Thread 0: `idx = 0 + 1*4 = 4`
- Block 2, Thread 3: `idx = 3 + 2*4 = 11`

### Why This Matters for MNIST

```cpp
__global__ void processImages(float* images, float* results, int numImages) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numImages) {  // Boundary check!
        // Thread idx processes image idx
        results[idx] = processOneImage(images + idx * 784);
    }
}
```

**Each thread gets its own image to process!**

### Exercise 3.1: Print Thread IDs

```cpp
__global__ void printIDs() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Block %d, Thread %d → Global ID %d\n", 
           blockIdx.x, threadIdx.x, idx);
}

int main() {
    printIDs<<<3, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

**Run it and verify you understand the pattern!**

---

# 🧱 **STEP 4 — Memory: The CPU↔GPU Dance**

### The Key Insight

**CPU memory and GPU memory are SEPARATE!**

```
CPU (Host)                    GPU (Device)
┌──────────────┐             ┌──────────────┐
│ Your data    │             │ GPU memory   │
│ [1,2,3,4,5]  │   ─copy→    │ [1,2,3,4,5]  │
│              │             │              │
│              │   ←copy─    │ [results]    │
└──────────────┘             └──────────────┘
```

You must **explicitly copy** data between them!

### The Memory Dance (7 Steps)

```cpp
// 1. Allocate CPU memory
float* h_data = new float[n];  // 'h_' = host (CPU)

// 2. Initialize data on CPU
for (int i = 0; i < n; i++) h_data[i] = i;

// 3. Allocate GPU memory
float* d_data;  // 'd_' = device (GPU)
cudaMalloc(&d_data, n * sizeof(float));

// 4. Copy data CPU → GPU
cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

// 5. Run kernel on GPU
myKernel<<<blocks, threads>>>(d_data, n);

// 6. Copy result GPU → CPU
cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

// 7. Free memory
cudaFree(d_data);
delete[] h_data;
```

### Memory Functions

| Function | What It Does |
|----------|--------------|
| `cudaMalloc(&ptr, size)` | Allocate on GPU |
| `cudaMemcpy(dst, src, size, direction)` | Copy data |
| `cudaFree(ptr)` | Free GPU memory |

**Direction options:**
- `cudaMemcpyHostToDevice` = CPU → GPU
- `cudaMemcpyDeviceToHost` = GPU → CPU
- `cudaMemcpyDeviceToDevice` = GPU → GPU

### Naming Convention

```cpp
float* h_weights;  // h_ = host (CPU)
float* d_weights;  // d_ = device (GPU)
```

**Always use this convention! It prevents bugs.**

---

# 🧱 **STEP 5 — Your First Useful Kernel: Vector Add**

### The Task

Add two arrays: `c[i] = a[i] + b[i]` for all i

### CPU Version (Phase 2 style)

```cpp
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];  // Sequential, one at a time
}
```

**Time for 1,000,000 elements:** ~1-2 ms

### GPU Kernel

```cpp
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {  // ALWAYS check bounds!
        c[idx] = a[idx] + b[idx];
    }
}
```

**Each thread handles ONE element!**

### Complete Program

```cpp
int main() {
    int n = 1000;
    int size = n * sizeof(float);
    
    // 1. Allocate and initialize host arrays
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];
    
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 2. Allocate device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 3. Copy input data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 4. Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // 5. Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 6. Verify results
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %.0f (expected %.0f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }
    
    // 7. Free memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    
    return 0;
}
```

### Why `(n + threadsPerBlock - 1) / threadsPerBlock`?

**Problem:** n might not divide evenly by threadsPerBlock

**Example:**
- n = 1000, threadsPerBlock = 256
- Need: ceil(1000 / 256) = ceil(3.906) = 4 blocks
- Formula: `(1000 + 256 - 1) / 256 = 1255 / 256 = 4` ✅

**This is the standard pattern - use it everywhere!**

### Exercise 5.1: Run Vector Add

Compile and run the program above. Verify all results are correct!

---

# 🧱 **STEP 6 — Porting Your Phase 2 Functions**

### ReLU on GPU

**Phase 2 (CPU):**
```cpp
vector<float> relu(const vector<float>& z) {
    vector<float> result(z.size());
    for (int i = 0; i < z.size(); i++) {
        result[i] = max(0.0f, z[i]);
    }
    return result;
}
```

**Phase 3 (GPU):**
```cpp
__global__ void reluKernel(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}
```

**What changed:**
- Loop → thread index
- `max()` → `fmaxf()` (GPU math function)
- Operates on raw pointers, not vectors

### Sigmoid on GPU

**Formula:** `sigmoid(x) = 1 / (1 + exp(-x))`

```cpp
__global__ void sigmoidKernel(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}
```

### Exercise 6.1: Test ReLU

```cpp
int main() {
    int n = 5;
    float h_input[] = {-2, -1, 0, 1, 2};
    float h_output[5];
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    reluKernel<<<1, n>>>(d_input, d_output, n);
    
    // Copy back
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify: should be [0, 0, 0, 1, 2]
    for (int i = 0; i < n; i++) {
        printf("ReLU(%.0f) = %.0f\n", h_input[i], h_output[i]);
    }
    
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
```

---

# 🧱 **STEP 7 — Matrix-Vector Multiplication**

### The Most Important Operation

**This is the core of your neural network!**

```
hiddenZ = W1 × input
outputZ = W2 × hiddenA
```

### CPU Version (Phase 2)

```cpp
// W1: 128×784, input: 784×1 → result: 128×1
for (int i = 0; i < 128; i++) {  // Each output element
    float sum = 0;
    for (int j = 0; j < 784; j++) {  // Dot product
        sum += W1[i][j] * input[j];
    }
    result[i] = sum;
}
```

### GPU Version

```cpp
__global__ void matVecMulKernel(float* matrix, float* vector, float* result,
                                 int rows, int cols) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}
```

**Key insight:** Each thread computes ONE output element (one row's dot product)

### Dimensions for MNIST

**Layer 1:**
```
W1: 128×784  (stored as flat array of 100,352 floats)
input: 784×1
result: 128×1

Launch: <<<(128 + 255)/256, 256>>>
→ 1 block, 256 threads (only 128 used)
```

**Layer 2:**
```
W2: 10×128
hiddenA: 128×1
result: 10×1

Launch: <<<1, 10>>>
→ 1 block, 10 threads
```

### Exercise 7.1: Test Matrix-Vector Multiply

```cpp
int main() {
    // Small test: 3×4 matrix × 4×1 vector = 3×1 result
    int rows = 3, cols = 4;
    
    float h_matrix[] = {
        1, 2, 3, 4,   // Row 0
        5, 6, 7, 8,   // Row 1
        9, 10, 11, 12 // Row 2
    };
    float h_vector[] = {1, 2, 3, 4};
    float h_result[3];
    
    // Expected: [30, 70, 110]
    // Row 0: 1*1 + 2*2 + 3*3 + 4*4 = 1+4+9+16 = 30
    // Row 1: 5*1 + 6*2 + 7*3 + 8*4 = 5+12+21+32 = 70
    // Row 2: 9*1 + 10*2 + 11*3 + 12*4 = 9+20+33+48 = 110
    
    // TODO: Allocate device memory, copy, launch kernel, verify
}
```

---

# 🧱 **STEP 8 — Error Checking (Critical!)**

### The Problem

CUDA errors fail **silently** by default!

```cpp
cudaMalloc(&d_data, size);  // Might fail, but code continues!
myKernel<<<blocks, threads>>>(d_data);  // Crashes mysteriously
```

### The Solution: Always Check Errors

**Helper macro:**
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

**Usage:**
```cpp
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

myKernel<<<blocks, threads>>>(d_data, n);
CUDA_CHECK(cudaGetLastError());  // Check kernel launch
CUDA_CHECK(cudaDeviceSynchronize());  // Check kernel execution
```

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `invalid argument` | Wrong parameters to kernel | Check dimensions |
| `out of memory` | cudaMalloc too large | Reduce batch size |
| `invalid device pointer` | Using CPU pointer on GPU | Check h_ vs d_ |
| `launch failed` | Kernel crashed | Add bounds checks |

**Add error checking to ALL your code!**

---

# 🧱 **STEP 9 — Putting It Together: Mini Neural Network**

### Goal

Run a simplified forward pass on GPU:
```
input (784) → W1 (128×784) → ReLU → W2 (10×128) → output (10)
```

### Complete Program Structure

```cpp
int main() {
    // 1. Load one MNIST image (on CPU)
    float h_input[784];  // Load from Phase 2
    
    // 2. Initialize random weights (on CPU)
    float h_W1[128 * 784];
    float h_W2[10 * 128];
    // ... initialize randomly ...
    
    // 3. Allocate GPU memory
    float *d_input, *d_W1, *d_W2, *d_hidden, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, 784 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, 128 * 784 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, 10 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 10 * sizeof(float)));
    
    // 4. Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 784 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1, 128 * 784 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2, 10 * 128 * sizeof(float), cudaMemcpyHostToDevice));
    
    // 5. Forward pass on GPU
    // Layer 1: W1 × input → hidden
    matVecMulKernel<<<1, 128>>>(d_W1, d_input, d_hidden, 128, 784);
    reluKernel<<<1, 128>>>(d_hidden, d_hidden, 128);
    
    // Layer 2: W2 × hidden → output
    matVecMulKernel<<<1, 10>>>(d_W2, d_hidden, d_output, 10, 128);
    
    // 6. Copy result back
    float h_output[10];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 10 * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 7. Print predictions
    for (int i = 0; i < 10; i++) {
        printf("Digit %d: %.4f\n", i, h_output[i]);
    }
    
    // 8. Free memory
    cudaFree(d_input); cudaFree(d_W1); cudaFree(d_W2);
    cudaFree(d_hidden); cudaFree(d_output);
    
    return 0;
}
```

### Exercise 9.1: Run It!

1. Use weights from your Phase 2 trained model
2. Run forward pass on GPU
3. Compare output with CPU version
4. They should match!

---

## ✅ **Phase 3 Checkpoint**

Before moving to Phase 4, you should be able to:

- [ ] Explain CPU vs GPU trade-offs
- [ ] Write CUDA kernels with `__global__`
- [ ] Compute global thread index correctly
- [ ] Allocate GPU memory with `cudaMalloc`
- [ ] Copy data between CPU and GPU
- [ ] Implement element-wise operations (ReLU, sigmoid)
- [ ] Implement matrix-vector multiplication
- [ ] Check for CUDA errors
- [ ] Run a simple forward pass on GPU

### Test Your Understanding

**Answer these without looking back:**

1. What does `threadIdx.x + blockIdx.x * blockDim.x` compute?
2. Why do we need `if (idx < n)` in kernels?
3. What's the difference between `cudaMalloc` and `malloc`?
4. How many threads in `<<<10, 256>>>`?
5. What does `cudaMemcpyHostToDevice` do?

**Answers:**
1. Global thread index
2. Might launch more threads than data elements
3. `cudaMalloc` allocates on GPU, `malloc` on CPU
4. 2,560 threads (10 blocks × 256 threads)
5. Copies data from CPU to GPU

---

## 🎯 **Next Steps**

**You now understand:**
- ✅ How GPUs work
- ✅ How to write CUDA kernels
- ✅ How to manage GPU memory
- ✅ How to port CPU code to GPU

**In Phase 4, you'll:**
- Port your complete neural network to GPU
- Implement backpropagation on GPU
- Train 10-50x faster than CPU
- Achieve same accuracy in seconds instead of minutes

**The hard part is done!** Phase 4 is just applying what you learned here to your Phase 2 code.

---

## 💡 **Key Takeaways**

**The CUDA Pattern:**
```
1. Allocate GPU memory
2. Copy data to GPU
3. Launch kernel (many threads)
4. Each thread: compute its index, process its data
5. Copy results back
6. Free GPU memory
```

**This pattern applies to EVERYTHING in Phase 4!**

**Congratulations!** You can now program GPUs! ⚡

Now let's use this power to accelerate your neural network!
