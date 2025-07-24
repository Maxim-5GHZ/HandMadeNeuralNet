#include "mlp.h"
#include <random>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>

template<typename T>
T MLP<T>::random_float(T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(min, max);
    return dist(gen);
}

template<typename T>
void MLP<T>::calculate() {
    for (size_t layer = 1; layer < data.size(); layer++) {
        for (size_t neuron = 0; neuron < data[layer].size(); neuron++) {
            T sum = bias[layer][neuron];
            for (size_t prev_neuron = 0; prev_neuron < data[layer - 1].size(); prev_neuron++) {
                sum += weights[layer - 1][prev_neuron][neuron] * data[layer - 1][prev_neuron];
            }
            data[layer][neuron] = sum;
        }

        auto& activation = activations[layer - 1];
        for (auto& val : data[layer]) {
            val = activation(val);
        }
    }
}

template<typename T>
MLP<T>::MLP(const std::vector<size_t>& neurons,
            const std::vector<typename Activator<T>::Function>& activate,
            T maxBiasValue) {
    Activator<T> activator(activate);
    activations = activator.getActivations();
    activationDerivatives = activator.getDerivatives();

    if (neurons.size() < 2) {
        throw std::invalid_argument("Network must have at least 2 layers");
    }

    if (activations.size() != neurons.size() - 1) {
        throw std::invalid_argument("Mismatch between layers and activations");
    }

    bias.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        bias[i].resize(neurons[i]);
        T co = T(0);
        for (auto& val : bias[i]) {
            val = std::sin(co * T(0.1)) * maxBiasValue;
            co += T(1);
        }
    }

    weights.resize(neurons.size() - 1);
    for (size_t i = 0; i < neurons.size() - 1; ++i) {
        T scale = std::sqrt(T(2) / static_cast<T>(neurons[i]));
        weights[i].resize(neurons[i]);
        for (size_t j = 0; j < neurons[i]; ++j) {
            weights[i][j].resize(neurons[i + 1]);
            for (size_t k = 0; k < neurons[i + 1]; ++k) {
                weights[i][j][k] = random_float(-scale, scale);
            }
        }
    }

    data.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        data[i].resize(neurons[i]);
    }
}

template<typename T>
std::vector<T> MLP<T>::train(const std::vector<T>& input,
                             const std::vector<T>& target,
                             T learning_rate) {
    if (input.size() != data[0].size()) {
        throw std::invalid_argument("Input size mismatch");
    }
    if (target.size() != data.back().size()) {
        throw std::invalid_argument("Target size mismatch");
    }

    data[0] = input;
    calculate();

    std::vector<std::vector<T>> gradients(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        gradients[i].resize(data[i].size(), T(0));
    }

    for (size_t i = 0; i < data.back().size(); ++i) {
        gradients.back()[i] = T(2) * (data.back()[i] - target[i]);
    }

    for (size_t layer = data.size() - 1; layer > 0; --layer) {
        auto& derivative = activationDerivatives[layer - 1];

        for (size_t neuron = 0; neuron < data[layer].size(); ++neuron) {
            T grad = gradients[layer][neuron] * derivative(data[layer][neuron]);
            grad = std::max(T(-1.0), std::min(T(1.0), grad));
            gradients[layer][neuron] = grad;

            for (size_t prev_neuron = 0; prev_neuron < data[layer - 1].size(); ++prev_neuron) {
                gradients[layer - 1][prev_neuron] += grad * weights[layer - 1][prev_neuron][neuron];
            }
        }
    }

    for (size_t layer = 0; layer < weights.size(); ++layer) {
        for (size_t j = 0; j < weights[layer].size(); ++j) {
            for (size_t k = 0; k < weights[layer][j].size(); ++k) {
                T delta = learning_rate * gradients[layer + 1][k] * data[layer][j];
                weights[layer][j][k] -= delta;
            }
        }
    }

    for (size_t layer = 1; layer < bias.size(); ++layer) {
        for (size_t neuron = 0; neuron < bias[layer].size(); ++neuron) {
            bias[layer][neuron] -= learning_rate * gradients[layer][neuron];
        }
    }

    return data.back();
}

template<typename T>
std::vector<T> MLP<T>::predict(const std::vector<T>& input) {
    if (input.size() != data[0].size()) {
        throw std::invalid_argument("Input size mismatch");
    }
    data[0] = input;
    calculate();
    return data.back();
}



template<typename T>
const std::vector<std::vector<std::vector<T>>>& MLP<T>::get_weights() const {
    return weights;
}


template<typename T>
const std::vector<std::vector<T>>& MLP<T>::get_biases() const {
    return bias;
}


template<typename T>
void MLP<T>::set_weights(const std::vector<std::vector<std::vector<T>>>& new_weights) {
    if (new_weights.size() != weights.size()) {
        throw std::invalid_argument("Invalid number of weight layers");
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        if (new_weights[i].size() != weights[i].size()) {
            throw std::invalid_argument("Invalid number of neurons in weight layer " + std::to_string(i));
        }
        
        for (size_t j = 0; j < weights[i].size(); ++j) {
            if (new_weights[i][j].size() != weights[i][j].size()) {
                throw std::invalid_argument("Invalid number of connections in layer " 
                    + std::to_string(i) + " neuron " + std::to_string(j));
            }
        }
    }
    
    weights = new_weights;
}


template<typename T>
void MLP<T>::set_biases(const std::vector<std::vector<T>>& new_biases) {
    if (new_biases.size() != bias.size()) {
        throw std::invalid_argument("Invalid number of bias layers");
    }
    
    for (size_t i = 0; i < bias.size(); ++i) {
        if (new_biases[i].size() != bias[i].size()) {
            throw std::invalid_argument("Invalid number of neurons in bias layer " + std::to_string(i));
        }
    }
    
    bias = new_biases;
}


template<typename T>
void MLP<T>::save_weights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for writing");

    size_t num_layers = bias.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    for (const auto& layer : bias) {
        size_t size = layer.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }

    for (const auto& layer : weights) {
        for (const auto& neuron_weights : layer) {
            file.write(reinterpret_cast<const char*>(neuron_weights.data()),
                       neuron_weights.size() * sizeof(T));
        }
    }

    for (const auto& layer : bias) {
        file.write(reinterpret_cast<const char*>(layer.data()),
                   layer.size() * sizeof(T));
    }
}

template<typename T>
void MLP<T>::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for reading");

    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (num_layers != bias.size()) {
        throw std::runtime_error("Network structure mismatch");
    }

    std::vector<size_t> layer_sizes(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (size != bias[i].size()) {
            throw std::runtime_error("Layer size mismatch");
        }
    }

    for (auto& layer : weights) {
        for (auto& neuron_weights : layer) {
            file.read(reinterpret_cast<char*>(neuron_weights.data()),
                      neuron_weights.size() * sizeof(T));
        }
    }

    for (auto& layer : bias) {
        file.read(reinterpret_cast<char*>(layer.data()),
                  layer.size() * sizeof(T));
    }
}


template class MLP<float>;
template class MLP<double>;