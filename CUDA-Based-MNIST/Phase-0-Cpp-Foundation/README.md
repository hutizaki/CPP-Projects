# 🧱 **Phase 0: C++ Foundation**

## 🎯 **Goal**

Master the C++ fundamentals required for CUDA programming: pointers, memory management, and low-level data structures.

**Time:** 2-4 days  
**Difficulty:** ⭐⭐

---

## 🤔 **Why This Phase Matters**

CUDA is an extension of C++. Before you can write GPU code, you need to be **completely comfortable** with:
- Pointers and pointer arithmetic
- Manual memory management (heap allocation)
- Understanding memory layouts
- Passing data between functions efficiently

If you're shaky on these concepts, CUDA will be confusing and frustrating. Master them here first.

---

## 📚 **What You'll Learn**

### **Module 1: Pointers & References**
- What pointers actually are (memory addresses)
- Pointer arithmetic
- Arrays vs pointers
- Pass-by-value vs pass-by-reference
- When to use each

### **Module 2: Memory Management**
- Stack vs heap
- `new` and `delete`
- `malloc` and `free`
- Memory leaks and how to avoid them
- RAII principle with `std::vector`

### **Module 3: Data Structures**
- Structs for organizing data
- Representing matrices with flat arrays
- Row-major vs column-major indexing
- Cache-friendly memory layouts

### **Module 4: Project Organization**
- Header files (`.h`) vs implementation files (`.cpp`)
- Include guards
- Separating interface from implementation
- Basic Makefile

---

## 🛠️ **Module 1: Pointers & References**

### **The Concept**

A pointer is a variable that stores a **memory address**.

```cpp
int x = 42;        // x is an integer
int* ptr = &x;     // ptr stores the address of x
int value = *ptr;  // value = 42 (dereference ptr)
```

**Visual:**
```
Memory:
Address  | Value
---------|-------
0x1000   | 42      ← x lives here
0x2000   | 0x1000  ← ptr stores x's address
```

### **Why Pointers Matter**

In CUDA, you'll constantly:
- Pass pointers to GPU memory
- Index into large arrays
- Manage memory manually

You need to be comfortable with this.

### **Pointer Arithmetic**

```cpp
int arr[5] = {10, 20, 30, 40, 50};
int* ptr = arr;  // Points to first element

ptr[0];     // 10
ptr[1];     // 20
*(ptr + 2); // 30 (pointer arithmetic)
```

**Key insight:** `ptr[i]` is the same as `*(ptr + i)`

### **Exercise 1.1: Pointer Basics**

Create `module1_pointers.cpp`:

```cpp
#include <iostream>
using namespace std;

int main() {
    // TODO: Declare an integer variable x with value 100
    
    // TODO: Declare a pointer ptr that points to x
    
    // TODO: Print the value of x using the pointer (dereference)
    
    // TODO: Change x to 200 using the pointer
    
    // TODO: Print x again to verify
    
    return 0;
}
```

**Expected output:**
```
x = 100
x = 200
```

### **Exercise 1.2: Array Traversal**

```cpp
#include <iostream>
using namespace std;

void printArray(int* arr, int size) {
    // TODO: Use pointer arithmetic to print all elements
    // Don't use [] notation, use *(arr + i)
}

int main() {
    int numbers[5] = {10, 20, 30, 40, 50};
    printArray(numbers, 5);
    return 0;
}
```

### **Exercise 1.3: Swap Function**

```cpp
// TODO: Implement a swap function using pointers
void swap(int* a, int* b) {
    // Your code here
}

int main() {
    int x = 5, y = 10;
    cout << "Before: x=" << x << ", y=" << y << endl;
    swap(&x, &y);
    cout << "After: x=" << x << ", y=" << y << endl;
    return 0;
}
```

---

## 🛠️ **Module 2: Memory Management**

### **The Concept: Stack vs Heap**

**Stack:**
- Automatic memory (declared in functions)
- Fast allocation
- Limited size
- Automatically cleaned up

```cpp
void foo() {
    int x = 42;  // On stack
}  // x is automatically destroyed here
```

**Heap:**
- Manual memory (you request it)
- Slower allocation
- Large size available
- YOU must clean it up

```cpp
void bar() {
    int* ptr = new int(42);  // On heap
    delete ptr;              // YOU must free it
}
```

### **Why Heap Matters for Neural Networks**

Neural network weights are LARGE:
- MNIST: 784 inputs × 128 hidden × 10 outputs = ~100,000 floats
- Can't fit on stack (typically ~8MB limit)
- Must use heap

### **Memory Leak Example**

```cpp
void leak() {
    int* data = new int[1000];
    // Forgot to delete!
}  // Memory is lost forever (until program ends)
```

### **RAII: Resource Acquisition Is Initialization**

Use `std::vector` to avoid manual memory management:

```cpp
#include <vector>

void safe() {
    std::vector<int> data(1000);
    // Automatically cleaned up when vector goes out of scope
}
```

### **Exercise 2.1: Dynamic Array**

```cpp
#include <iostream>
using namespace std;

int main() {
    int size;
    cout << "Enter array size: ";
    cin >> size;
    
    // TODO: Allocate array of 'size' floats on heap
    
    // TODO: Fill array with values 0, 1, 2, ...
    
    // TODO: Print the array
    
    // TODO: Free the memory
    
    return 0;
}
```

### **Exercise 2.2: Matrix Allocation**

```cpp
// Allocate a 2D matrix as a flat 1D array
float* allocateMatrix(int rows, int cols) {
    // TODO: Allocate rows * cols floats
    // Return pointer
}

void freeMatrix(float* matrix) {
    // TODO: Free the memory
}

// Access element at (row, col)
float getElement(float* matrix, int cols, int row, int col) {
    // TODO: Return matrix[row * cols + col]
}

void setElement(float* matrix, int cols, int row, int col, float value) {
    // TODO: Set matrix[row * cols + col] = value
}
```

### **Exercise 2.3: Vector vs Raw Pointer**

Implement the same functionality twice:

```cpp
// Version 1: Using std::vector
void testVector() {
    vector<float> data(1000);
    // Fill and use data
    // Automatically cleaned up
}

// Version 2: Using raw pointers
void testRawPointer() {
    float* data = new float[1000];
    // Fill and use data
    delete[] data;  // Must remember this!
}
```

**Question:** Which is safer? Which is more common in CUDA code?

---

## 🛠️ **Module 3: Data Structures**

### **The Concept: Flat Arrays for Matrices**

Matrices in memory are stored as **flat 1D arrays**.

**Row-major order** (C/C++ default):
```
Matrix:
[1 2 3]
[4 5 6]

Memory: [1, 2, 3, 4, 5, 6]
```

**Index formula:** `matrix[row * num_cols + col]`

### **Why This Matters**

GPU memory is linear. You need to:
- Store matrices as flat arrays
- Convert 2D indices to 1D indices
- Understand memory layout for performance

### **Exercise 3.1: Matrix Struct**

```cpp
struct Matrix {
    float* data;  // Pointer to heap-allocated array
    int rows;
    int cols;
    
    // Constructor: allocate memory
    Matrix(int r, int c) {
        // TODO: Implement
    }
    
    // Destructor: free memory
    ~Matrix() {
        // TODO: Implement
    }
    
    // Get element at (row, col)
    float get(int row, int col) {
        // TODO: Implement
    }
    
    // Set element at (row, col)
    void set(int row, int col, float value) {
        // TODO: Implement
    }
    
    // Print matrix
    void print() {
        // TODO: Implement
    }
};
```

### **Exercise 3.2: Matrix Operations**

```cpp
// Fill matrix with zeros
void zeros(Matrix& m) {
    // TODO: Implement
}

// Fill matrix with random values
void randomize(Matrix& m) {
    // TODO: Implement
}

// Matrix-vector multiply: y = M * x
void matVecMul(Matrix& M, float* x, float* y) {
    // TODO: Implement
    // y[i] = sum of M[i][j] * x[j] for all j
}
```

### **Exercise 3.3: Memory Layout Exploration**

```cpp
int main() {
    Matrix m(3, 4);  // 3 rows, 4 cols
    
    // Fill with sequential values
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            m.set(i, j, i * 4 + j);
        }
    }
    
    // Print as matrix
    m.print();
    
    // Print raw memory
    cout << "\nRaw memory: ";
    for (int i = 0; i < 12; i++) {
        cout << m.data[i] << " ";
    }
    cout << endl;
    
    return 0;
}
```

**Expected output:**
```
Matrix:
0  1  2  3
4  5  6  7
8  9  10 11

Raw memory: 0 1 2 3 4 5 6 7 8 9 10 11
```

---

## 🛠️ **Module 4: Project Organization**

### **The Concept: Separation of Interface and Implementation**

**Header file (`.h`):** Declarations (what exists)
**Source file (`.cpp`):** Definitions (how it works)

### **Example: Matrix.h**

```cpp
#ifndef MATRIX_H
#define MATRIX_H

struct Matrix {
    float* data;
    int rows;
    int cols;
    
    Matrix(int r, int c);
    ~Matrix();
    
    float get(int row, int col);
    void set(int row, int col, float value);
    void print();
};

// Function declarations
void zeros(Matrix& m);
void randomize(Matrix& m);
void matVecMul(Matrix& M, float* x, float* y);

#endif
```

### **Example: Matrix.cpp**

```cpp
#include "Matrix.h"
#include <iostream>
#include <cstdlib>

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data = new float[rows * cols];
}

Matrix::~Matrix() {
    delete[] data;
}

// ... implementations of other functions ...
```

### **Example: main.cpp**

```cpp
#include "Matrix.h"
#include <iostream>

int main() {
    Matrix m(3, 3);
    zeros(m);
    m.print();
    return 0;
}
```

### **Exercise 4.1: Create a Multi-File Project**

Create these files:
1. `Vector.h` - Vector struct declaration
2. `Vector.cpp` - Vector implementation
3. `main.cpp` - Test program

**Vector.h:**
```cpp
#ifndef VECTOR_H
#define VECTOR_H

struct Vector {
    float* data;
    int size;
    
    Vector(int s);
    ~Vector();
    
    float get(int i);
    void set(int i, float value);
    void print();
};

// Dot product
float dot(Vector& a, Vector& b);

#endif
```

Implement the rest yourself!

### **Exercise 4.2: Write a Makefile**

```makefile
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

all: main

main: main.o Vector.o
	$(CXX) $(CXXFLAGS) -o main main.o Vector.o

main.o: main.cpp Vector.h
	$(CXX) $(CXXFLAGS) -c main.cpp

Vector.o: Vector.cpp Vector.h
	$(CXX) $(CXXFLAGS) -c Vector.cpp

clean:
	rm -f *.o main
```

---

## ✅ **Phase 0 Checkpoint**

Before moving to Phase 1, you should be able to:

- [ ] Explain what a pointer is and when to use one
- [ ] Allocate and free heap memory without leaks
- [ ] Implement a Matrix struct with proper memory management
- [ ] Convert 2D indices to 1D array indices
- [ ] Organize code into header and source files
- [ ] Write a Makefile to build multi-file projects
- [ ] Implement matrix-vector multiplication

### **Final Project: Complete Matrix Library**

Create a small matrix library with:

**Files:**
- `Matrix.h` / `Matrix.cpp`
- `operations.h` / `operations.cpp` (matrix operations)
- `test.cpp` (test program)
- `Makefile`

**Features:**
- Matrix creation, destruction
- Element access
- Matrix-vector multiply
- Matrix-matrix multiply (optional)
- Print function
- Zero/random initialization

**Test it:**
```cpp
int main() {
    Matrix A(3, 4);
    randomize(A);
    
    Vector x(4);
    for (int i = 0; i < 4; i++) {
        x.set(i, 1.0);
    }
    
    Vector y(3);
    matVecMul(A, x, y);
    
    cout << "A:" << endl;
    A.print();
    cout << "x:" << endl;
    x.print();
    cout << "y = A*x:" << endl;
    y.print();
    
    return 0;
}
```

---

## 📚 **Additional Resources**

### **Recommended Reading:**
- C++ Primer (Chapters on pointers and memory)
- Effective C++ by Scott Meyers (Items on memory management)

### **Practice Problems:**
- Implement a dynamic array class
- Write a memory leak detector
- Implement a simple string class

---

## 🎯 **Next Steps**

Once you've completed the checkpoint:

1. ✅ Save your Matrix library - you'll use it in Phase 2
2. ✅ Make sure you understand memory layouts
3. ✅ Move to **Phase 1: Neural Network Math**

**Congratulations!** You now have the C++ foundation needed for CUDA programming! 🎉

