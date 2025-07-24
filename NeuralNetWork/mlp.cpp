#include "mlp.h"

template<typename T>
MLP<T>::MLP(const std::vector<size_t>& neurons,
    const std::vector<typename Activator<T>::Function>& activate,
    T maxBiasValue)
    : Perceptrone<T>(neurons, activate, maxBiasValue) {}

template<typename T>
std::vector<T> MLP<T>::train(const std::vector<T>& input,
                   const std::vector<T>& target,
                   T learning_rate) {
    if (input.size() != this->data[0].size()) {
        throw std::invalid_argument("Input size mismatch");
    }
    if (target.size() != this->data.back().size()) {
        throw std::invalid_argument("Target size mismatch");
    }

    this->data[0] = input;
    this->calculate();

    std::vector<std::vector<T>> gradients(this->data.size());
    for (size_t i = 0; i < this->data.size(); ++i) {
        gradients[i].resize(this->data[i].size(), T(0));
    }

    for (size_t i = 0; i < this->data.back().size(); ++i) {
        gradients.back()[i] = T(2) * (this->data.back()[i] - target[i]);
    }

    for (size_t layer = this->data.size() - 1; layer > 0; --layer) {
        auto& derivative = this->activationDerivatives[layer - 1];

        for (size_t neuron = 0; neuron < this->data[layer].size(); ++neuron) {
            T grad = gradients[layer][neuron] * derivative(this->data[layer][neuron]);
            grad = std::max(T(-1.0), std::min(T(1.0), grad));
            gradients[layer][neuron] = grad;

            for (size_t prev_neuron = 0; prev_neuron < this->data[layer - 1].size(); ++prev_neuron) {
                gradients[layer - 1][prev_neuron] += grad * this->weights[layer - 1][prev_neuron][neuron];
            }
        }
    }

    for (size_t layer = 0; layer < this->weights.size(); ++layer) {
        for (size_t j = 0; j < this->weights[layer].size(); ++j) {
            for (size_t k = 0; k < this->weights[layer][j].size(); ++k) {
                T delta = learning_rate * gradients[layer + 1][k] * this->data[layer][j];
                this->weights[layer][j][k] -= delta;
            }
        }
    }

    for (size_t layer = 1; layer < this->bias.size(); ++layer) {
        for (size_t neuron = 0; neuron < this->bias[layer].size(); ++neuron) {
            this->bias[layer][neuron] -= learning_rate * gradients[layer][neuron];
        }
    }

    return this->data.back();
}

template class MLP<float>;
template class MLP<double>;