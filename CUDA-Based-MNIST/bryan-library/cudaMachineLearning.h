#ifndef CUDAMACHINELEARNING_H
#define CUDAMACHINELEARNING_H

#include <vector>

// Forward declaration
namespace cudaMM {
    class GPUWeights;
}

namespace cudaML {
    /// @brief CUDA-accelerated neural network layer processing
    /// @param weights Weight matrix [num_neurons x input_size] where each row represents one neuron's weights
    /// @param bias Bias vector [num_neurons] containing the bias term for each neuron
    /// @param input Input vector [input_size] containing the input values to the layer
    /// @return Output vector [num_neurons] containing the pre-activation values (z = W*x + b)
    /// @details Computes the linear transformation z = W*x + b for a neural network layer.
    ///          Performs matrix-vector multiplication (weights * input) then adds the bias vector.
    ///          This is the pre-activation output; apply an activation function (sigmoid, ReLU, etc.) separately.
    /// @throws std::runtime_error if dimensions don't match (weights columns != input size, or weights rows != bias size)
    std::vector<float> processLayer(
        const std::vector<std::vector<float>>& weights, 
        const std::vector<float>& bias, 
        const std::vector<float>& input);

    /// @brief Batched forward pass on GPU - processes entire batch at once
    /// @param weights GPU weights manager (from cudaMM namespace)
    /// @param d_batch_input Device pointer to batch input [batch_size x input_dim]
    /// @param batch_size Number of samples in batch
    /// @param d_hiddenZ Output: hidden layer pre-activation (device pointer)
    /// @param d_hiddenA Output: hidden layer activation (device pointer)
    /// @param d_outputZ Output: output layer pre-activation (device pointer)
    void batchedForwardPass(cudaMM::GPUWeights& weights, float* d_batch_input, int batch_size,
                            float* d_hiddenZ, float* d_hiddenA, float* d_outputZ);

    /// @brief Batched backward pass on GPU - computes gradients and updates weights
    /// @param weights GPU weights manager (weights will be updated in-place)
    /// @param d_batch_input Device pointer to batch input [batch_size x input_dim]
    /// @param d_trueLabels Device pointer to true labels [batch_size]
    /// @param d_hiddenZ Device pointer to hidden layer pre-activation [batch_size x hidden_dim]
    /// @param d_hiddenA Device pointer to hidden layer activation [batch_size x hidden_dim]
    /// @param d_outputZ Device pointer to output layer pre-activation [batch_size x output_dim]
    /// @param d_y_hat Device pointer to softmax output [batch_size x output_dim] (will be computed)
    /// @param d_dW1 Device pointer for W1 gradients [hidden_dim x input_dim] (temporary, will be zeroed)
    /// @param d_db1 Device pointer for b1 gradients [hidden_dim] (temporary, will be zeroed)
    /// @param d_dW2 Device pointer for W2 gradients [output_dim x hidden_dim] (temporary, will be zeroed)
    /// @param d_db2 Device pointer for b2 gradients [output_dim] (temporary, will be zeroed)
    /// @param batch_size Number of samples in batch
    /// @param learning_rate Learning rate for weight updates
    void batchedBackwardPass(cudaMM::GPUWeights& weights,
                            float* d_batch_input, int* d_trueLabels,
                            float* d_hiddenZ, float* d_hiddenA, float* d_outputZ, float* d_y_hat,
                            float* d_dW1, float* d_db1, float* d_dW2, float* d_db2,
                            int batch_size, float learning_rate);
}

#endif // CUDAMACHINELEARNING_H