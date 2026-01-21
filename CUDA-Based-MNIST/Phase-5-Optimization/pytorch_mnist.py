"""
PyTorch MNIST Implementation
Matches Phase 4 architecture and training parameters
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from pathlib import Path
from mnist_loader import load_mnist_images, load_mnist_labels
from load_weights import load_initial_weights

def categorical_cross_entropy(y_pred, y_true):
    """Compute categorical cross-entropy loss"""
    # y_pred: [batch_size, 10] probabilities
    # y_true: [batch_size] class indices
    batch_size = y_pred.size(0)
    log_probs = torch.log(y_pred + 1e-8)  # Add small epsilon for numerical stability
    loss = -torch.sum(log_probs[range(batch_size), y_true]) / batch_size
    return loss.item()

def train_mnist_pytorch(train_images_path, train_labels_path, 
                        test_images_path, test_labels_path,
                        num_neurons=128, epochs=10, learning_rate=0.1, batch_size=32,
                        initial_weights_path=None):
    """
    Train MNIST classifier using PyTorch
    
    Returns:
        tuple: (training_time, test_accuracy, model)
    """
    print("=== PyTorch MNIST Training ===")
    print(f"Architecture: 784 -> {num_neurons} (ReLU) -> 10 (Softmax)")
    print(f"Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}\n")
    
    # Load data
    print("Loading training data...")
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    
    print("\nLoading test data...")
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)
    
    # Convert to PyTorch tensors
    train_images_tensor = torch.FloatTensor(train_images)
    train_labels_tensor = torch.LongTensor(train_labels)
    test_images_tensor = torch.FloatTensor(test_images)
    test_labels_tensor = torch.LongTensor(test_labels)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Move data to device
    train_images_tensor = train_images_tensor.to(device)
    train_labels_tensor = train_labels_tensor.to(device)
    test_images_tensor = test_images_tensor.to(device)
    test_labels_tensor = test_labels_tensor.to(device)
    
    # Define model (matching Phase 4 architecture)
    model = nn.Sequential(
        nn.Linear(784, num_neurons),
        nn.ReLU(),
        nn.Linear(num_neurons, 10),
        nn.Softmax(dim=1)
    ).to(device)
    
    # Load initial weights if provided, otherwise use random initialization
    if initial_weights_path and os.path.exists(f"{initial_weights_path}_W1.bin"):
        print(f"Loading initial weights from {initial_weights_path}_*.bin")
        try:
            w1, w2, b1, b2 = load_initial_weights(initial_weights_path)
            
            # Set weights and biases (make arrays writable to avoid warning)
            with torch.no_grad():
                model[0].weight.data = torch.FloatTensor(w1.copy()).to(device)
                model[0].bias.data = torch.FloatTensor(b1.copy()).to(device)
                model[2].weight.data = torch.FloatTensor(w2.copy()).to(device)
                model[2].bias.data = torch.FloatTensor(b2.copy()).to(device)
            
            print("Initial weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load initial weights: {e}")
            print("Using random initialization instead")
            # Xavier initialization as fallback
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)
    else:
        # Use Xavier/Glorot initialization (matching C++ random range approximately)
        print("Using random initialization (Xavier)")
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
    
    # Optimizer (SGD matching Phase 4)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n=== Training ===")
    
    # Synchronize before timing to ensure all GPU operations are ready
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle data (on GPU to keep data on device, but adds overhead)
        # Note: GPU randperm for 60k elements can take 50-100ms per epoch
        indices = torch.randperm(len(train_images_tensor), device=device)
        train_images_shuffled = train_images_tensor[indices]
        train_labels_shuffled = train_labels_tensor[indices]
        
        # Process in batches
        for batch_start in range(0, len(train_images_tensor), batch_size):
            batch_end = min(batch_start + batch_size, len(train_images_tensor))
            batch_images = train_images_shuffled[batch_start:batch_end]
            batch_labels = train_labels_shuffled[batch_start:batch_end]
            
            # Forward pass
            outputs = model(batch_images)
            
            # Backward pass
            optimizer.zero_grad()
            # Compute loss tensor for backpropagation (keep on GPU, no sync)
            log_probs = torch.log(outputs + 1e-8)
            loss_tensor = -torch.sum(log_probs[range(len(batch_images)), batch_labels]) / len(batch_images)
            loss_tensor.backward()
            optimizer.step()
            
            # Compute loss for logging (this triggers CPU-GPU sync, but only for logging)
            # Note: This adds overhead but is necessary for epoch loss reporting
            # We do this AFTER backward pass to minimize impact
            loss_value = loss_tensor.item()  # Single sync per batch (faster than categorical_cross_entropy)
            total_loss += loss_value * len(batch_images)
            num_batches += 1
        
        # Compute average loss (single CPU operation)
        avg_loss = total_loss / len(train_images_tensor)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Synchronize after training to ensure all GPU operations complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    training_time = end_time - start_time
    
    # Test accuracy
    print("\n=== Testing ===")
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_images_tensor)
        predictions = torch.argmax(test_outputs, dim=1)
        correct = (predictions == test_labels_tensor).sum().item()
        accuracy = (correct / len(test_labels_tensor)) * 100.0
    
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{len(test_labels_tensor)})")
    
    return training_time, accuracy, model

if __name__ == "__main__":
    # Paths relative to Phase-5-Optimization directory
    train_images_path = "../train-images.idx3-ubyte"
    train_labels_path = "../train-labels.idx1-ubyte"
    test_images_path = "../t10k-images.idx3-ubyte"
    test_labels_path = "../t10k-labels.idx1-ubyte"
    
    # Try to load initial weights if they exist
    initial_weights_path = "initial_weights"
    
    training_time, accuracy, model = train_mnist_pytorch(
        train_images_path, train_labels_path,
        test_images_path, test_labels_path,
        initial_weights_path=initial_weights_path
    )
    
    print(f"\n=== Results ===")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Test Accuracy: {accuracy:.2f}%")
