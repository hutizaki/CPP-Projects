# Phase 1 — Neural Network Math

## Goal

Understand the mathematical foundations of neural networks so you can implement one **from scratch in C++** (no ML libraries).

* **Time:** ~4–6 days
* **Difficulty:** Medium

---

## What Changed From Phase 0

### Phase 0 — Single Neuron

```
z = w·x + b
y = sigmoid(z)
```

This is a dot product plus a bias.

### Phase 1 — Layers of Neurons

```
z = W × x + b
a = activation(z)
```

* One neuron → dot product
* Multiple neurons → matrix × vector

This is the jump from scalars to linear algebra.

---

## Practice Problem — XOR

```
x1 x2 y
0  0  0
0  1  1
1  0  1
1  1  0
```

**Why XOR?**

* A single neuron cannot learn XOR
* Requires a hidden layer
* Forces correct use of matrix math and activation functions

If you can explain *why* XOR needs multiple neurons, you’re on track.

---

## Step 1 — Vectors & Matrices

* Vector = list of numbers
* Dot product = multiply corresponding elements, then sum
* Matrix × vector = many dot products at once

**Shape rule**

```
(m × k) × (k × n) = (m × n)
```

Each row of the matrix represents **one neuron**.

---

## Step 2 — Forward Pass

A layer computes:

```
z = W × x + b
a = activation(z)
```

**Common activations**

* **ReLU**: `max(0, z)` → hidden layers
* **Sigmoid**: outputs `[0, 1]` → binary classification
* **Softmax**: probabilities that sum to 1 → multi-class output

Without activation functions, deep networks collapse into linear models.

---

## Step 3 — Loss (Measuring Error)

Loss answers one question:

> **How wrong was the prediction?**

For classification, use **cross-entropy loss**:

```
loss = -log(probability of the correct class)
```

* Correct + confident → small loss
* Wrong + confident → very large loss

---

## Step 4 — Backpropagation

Backpropagation is **just the chain rule**.

For a weight `w`:

```
dL/dw = (dL/da) × (da/dz) × (dz/dw)
```

Key derivatives:

* **ReLU′** = 1 if `z > 0`, else 0
* **Sigmoid′** = `a × (1 - a)`

Gradients flow **backward** from loss to weights.

---

## Step 5 — Learning (Gradient Descent)

Update rule:

```
w = w - learning_rate × gradient
```

* Gradient points uphill (worse loss)
* Minus sign moves downhill (better loss)

Learning rate:

* Too large → unstable
* Too small → slow
* Just right → steady improvement

---

## Phase 1 Checklist

### Core Concepts

* [ ] I understand what a vector is
* [ ] I understand what a matrix is
* [ ] I can compute a dot product by hand
* [ ] I can do matrix × vector multiplication
* [ ] I understand matrix shapes and why dimensions must match

### Neurons & Layers

* [ ] I know a neuron computes `z = w·x + b`
* [ ] I understand a layer computes `z = W × x + b`
* [ ] I know each row of `W` represents one neuron
* [ ] I know why bias is required

### Forward Pass

* [ ] I can compute a forward pass for one neuron
* [ ] I can compute a forward pass for a layer
* [ ] I understand what an activation function does
* [ ] I know why activations are required (nonlinearity)
* [ ] I know when to use ReLU vs Sigmoid vs Softmax

### Loss (Error Measurement)

* [ ] I understand loss is a single number measuring wrongness
* [ ] I know cross-entropy is used for classification
* [ ] I understand `loss = -log(probability of the correct class)`
* [ ] I know confident wrong predictions give high loss
* [ ] I know confident correct predictions give low loss
* [ ] I can compute cross-entropy loss by hand

### Backpropagation

* [ ] I understand backprop is just the chain rule
* [ ] I know the gradient form `dL/dw = (dL/da)(da/dz)(dz/dw)`
* [ ] I know the derivative of ReLU
* [ ] I know the derivative of sigmoid
* [ ] I can compute gradients for a single neuron
* [ ] I understand gradient flow through layers

### Gradient Descent

* [ ] I know the update rule `w = w - lr × gradient`
* [ ] I understand why the minus sign is required
* [ ] I know what the learning rate controls
* [ ] I can manually perform one gradient descent step
* [ ] I know common learning rate failure cases

### XOR Understanding

* [ ] I know why a single neuron cannot learn XOR
* [ ] I understand why a hidden layer is required
* [ ] I understand how multiple neurons create non-linear boundaries

### Ready for Phase 2

* [ ] I can explain the forward pass in my own words
* [ ] I can explain backpropagation in my own words
* [ ] I can explain cross-entropy loss without notes
* [ ] I feel confident implementing this in C++ without ML libraries

---

## Next Phase

**Phase 2 — CPU Neural Network in C++**

You will manually implement:

* Matrices
* Forward pass
* Backpropagation
* Gradient descent

If you can complete this checklist honestly, you are ready.
