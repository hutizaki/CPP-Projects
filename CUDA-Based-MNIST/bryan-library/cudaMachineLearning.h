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
}

#endif // CUDAMACHINELEARNING_H