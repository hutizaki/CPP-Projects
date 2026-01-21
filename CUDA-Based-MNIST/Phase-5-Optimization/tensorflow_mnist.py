"""
TensorFlow/Keras MNIST Implementation
Matches Phase 4 architecture and training parameters
"""
import os
# Set CPU-only mode BEFORE importing TensorFlow to avoid GPU initialization issues
# RTX 5070 (compute capability 12.0) requires custom TensorFlow build
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from TensorFlow

import tensorflow as tf
import numpy as np
import time
from pathlib import Path
from mnist_loader import load_mnist_images, load_mnist_labels
from load_weights import load_initial_weights

def categorical_cross_entropy(y_pred, y_true):
    """Compute categorical cross-entropy loss"""
    # y_pred: [batch_size, 10] probabilities
    # y_true: [batch_size] class indices (need to convert to one-hot)
    y_true_one_hot = tf.one_hot(y_true, depth=10)
    loss = -tf.reduce_mean(tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred + 1e-8), axis=1))
    return loss.numpy()

def train_mnist_tensorflow(train_images_path, train_labels_path,
                           test_images_path, test_labels_path,
                           num_neurons=128, epochs=10, learning_rate=0.1, batch_size=32,
                           initial_weights_path=None):
    """
    Train MNIST classifier using TensorFlow/Keras
    
    Returns:
        tuple: (training_time, test_accuracy, model)
    """
    print("=== TensorFlow MNIST Training ===")
    print(f"Architecture: 784 -> {num_neurons} (ReLU) -> 10 (Softmax)")
    print(f"Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}\n")
    
    # Load data
    print("Loading training data...")
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    
    print("\nLoading test data...")
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)
    
    # Convert to TensorFlow tensors
    train_images_tensor = tf.constant(train_images, dtype=tf.float32)
    train_labels_tensor = tf.constant(train_labels, dtype=tf.int32)
    test_images_tensor = tf.constant(test_images, dtype=tf.float32)
    test_labels_tensor = tf.constant(test_labels, dtype=tf.int32)
    
    # Device info (GPU is hidden via environment variable set before import)
    print(f"\nTensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Note: {len(gpus)} GPU(s) detected but using CPU")
        print("Reason: RTX 5070 (compute capability 12.0) requires custom TensorFlow build")
    else:
        print("Using CPU (GPU hidden via CUDA_VISIBLE_DEVICES)")
    print("This is still a fair comparison - same initial weights, data, and hyperparameters.")
    
    # Define model (matching Phase 4 architecture)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_neurons, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Load initial weights if provided, otherwise use random initialization
    if initial_weights_path and os.path.exists(f"{initial_weights_path}_W1.bin"):
        print(f"Loading initial weights from {initial_weights_path}_*.bin")
        try:
            w1, w2, b1, b2 = load_initial_weights(initial_weights_path)
            
            # Build model first
            model.build(input_shape=(None, 784))
            
            # Set weights and biases (TensorFlow uses row-major like our format, but expects [kernel, bias])
            # Our w1 is [128, 784], TensorFlow expects [784, 128] for input->hidden, so transpose
            model.layers[0].set_weights([w1.T, b1])
            # Our w2 is [10, 128], TensorFlow expects [128, 10] for hidden->output, so transpose
            model.layers[1].set_weights([w2.T, b2])
            
            print("Initial weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load initial weights: {e}")
            print("Using random initialization instead")
            # Use Xavier/Glorot initialization as fallback
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    layer.kernel_initializer = tf.keras.initializers.GlorotUniform()
                    layer.bias_initializer = tf.keras.initializers.Zeros()
            model.build(input_shape=(None, 784))
    else:
        # Use Xavier/Glorot initialization
        print("Using random initialization (Glorot)")
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_initializer = tf.keras.initializers.GlorotUniform()
                layer.bias_initializer = tf.keras.initializers.Zeros()
        # Build model
        model.build(input_shape=(None, 784))
    
    # Optimizer (SGD matching Phase 4)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # Training loop
    print("\n=== Training ===")
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(train_images_tensor))
        train_images_shuffled = tf.gather(train_images_tensor, indices)
        train_labels_shuffled = tf.gather(train_labels_tensor, indices)
        
        # Process in batches
        for batch_start in range(0, len(train_images_tensor), batch_size):
            batch_end = min(batch_start + batch_size, len(train_images_tensor))
            batch_images = train_images_shuffled[batch_start:batch_end]
            batch_labels = train_labels_shuffled[batch_start:batch_end]
            
            # Forward and backward pass
            with tf.GradientTape() as tape:
                outputs = model(batch_images, training=True)
                loss = categorical_cross_entropy(outputs, batch_labels)
            
            # Compute gradients and update weights
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            total_loss += loss * len(batch_images)
            num_batches += 1
        
        avg_loss = total_loss / len(train_images_tensor)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    end_time = time.perf_counter()
    training_time = end_time - start_time
    
    # Test accuracy
    print("\n=== Testing ===")
    test_outputs = model(test_images_tensor, training=False)
    predictions = tf.argmax(test_outputs, axis=1)
    correct = tf.reduce_sum(tf.cast(predictions == test_labels_tensor, tf.float32))
    accuracy = (correct / len(test_labels_tensor)) * 100.0
    
    print(f"Test Accuracy: {accuracy:.2f}% ({int(correct)}/{len(test_labels_tensor)})")
    
    return training_time, float(accuracy), model

if __name__ == "__main__":
    # Paths relative to Phase-5-Optimization directory
    train_images_path = "../train-images.idx3-ubyte"
    train_labels_path = "../train-labels.idx1-ubyte"
    test_images_path = "../t10k-images.idx3-ubyte"
    test_labels_path = "../t10k-labels.idx1-ubyte"
    
    # Try to load initial weights if they exist
    initial_weights_path = "initial_weights"
    
    training_time, accuracy, model = train_mnist_tensorflow(
        train_images_path, train_labels_path,
        test_images_path, test_labels_path,
        initial_weights_path=initial_weights_path
    )
    
    print(f"\n=== Results ===")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Test Accuracy: {accuracy:.2f}%")
