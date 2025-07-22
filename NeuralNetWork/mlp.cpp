#include "mlp.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

using namespace std;

double MLP::random_double(double min, double max) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dist(min, max);
    return dist(gen);
}


MLP::MLP(const vector<size_t>& neurons, 
        const vector<Activator::Function>& activate,
        double maxBiasValue) {
    
    Activator activator(activate);
    activations = activator.getActivations();
    activationDerivatives = activator.getDerivatives();

    if (neurons.size() < 2) {
        throw invalid_argument("Network must have at least 2 layers");
    }

    if (activations.size() != neurons.size() - 1) {
        throw invalid_argument("Mismatch between layers and activations");
    }


    bias.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        bias[i].resize(neurons[i]);
        double co;
        for (auto& val : bias[i]) {
            co++;
            val = sin(co*0.1)*maxBiasValue;
        }
    }

    
    weights.resize(neurons.size() - 1);
    for (size_t i = 0; i < neurons.size() - 1; ++i) {
        double scale = sqrt(2.0 / neurons[i]);
        weights[i].resize(neurons[i]);
        for (size_t j = 0; j < neurons[i]; ++j) {
            weights[i][j].resize(neurons[i + 1]);
            for (size_t k = 0; k < neurons[i + 1]; ++k) {
                weights[i][j][k] = random_double(-scale, scale);
            }
        }
    }

    
    data.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        data[i].resize(neurons[i]);
    }
}

void MLP::calculate() {
    for (size_t layer = 1; layer < data.size(); layer++) {
        for (size_t neuron = 0; neuron < data[layer].size(); neuron++) {
            double sum = bias[layer][neuron];
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

vector<double> MLP::train(const vector<double>& input, 
                         const vector<double>& target, 
                         double learning_rate) {
    if (input.size() != data[0].size()) {
        throw invalid_argument("Input size mismatch");
    }
    if (target.size() != data.back().size()) {
        throw invalid_argument("Target size mismatch");
    }

    
    data[0] = input;
    calculate();

    
    vector<vector<double>> gradients(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        gradients[i].resize(data[i].size(), 0.0);
    }

    
    for (size_t i = 0; i < data.back().size(); ++i) {
        gradients.back()[i] = 2.0 * (data.back()[i] - target[i]);
    }

    
    for (size_t layer = data.size() - 1; layer > 0; --layer) {
        auto& derivative = activationDerivatives[layer - 1];

        for (size_t neuron = 0; neuron < data[layer].size(); ++neuron) {
            double grad = gradients[layer][neuron] * derivative(data[layer][neuron]);
            
            
            grad = max(-1.0, min(1.0, grad));
            gradients[layer][neuron] = grad;

            
            for (size_t prev_neuron = 0; prev_neuron < data[layer - 1].size(); ++prev_neuron) {
                gradients[layer - 1][prev_neuron] += grad * weights[layer - 1][prev_neuron][neuron];
            }
        }
    }

    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        for (size_t j = 0; j < weights[layer].size(); ++j) {
            for (size_t k = 0; k < weights[layer][j].size(); ++k) {
                double delta = learning_rate * gradients[layer+1][k] * data[layer][j];
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

vector<double> MLP::predict(const vector<double>& input) {
    if (input.size() != data[0].size()) {
        throw invalid_argument("Input size mismatch");
    }
    data[0] = input;
    calculate();
    return data.back();
}

void MLP::save_weights(const string& filename) const {
    ofstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open file for writing");

    
    size_t num_layers = bias.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    for (const auto& layer : bias) {
        size_t size = layer.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }

    
    for (const auto& layer : weights) {
        for (const auto& neuron_weights : layer) {
            file.write(reinterpret_cast<const char*>(neuron_weights.data()),
                       neuron_weights.size() * sizeof(double));
        }
    }

   
    for (const auto& layer : bias) {
        file.write(reinterpret_cast<const char*>(layer.data()),
                   layer.size() * sizeof(double));
    }
}

void MLP::load_weights(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open file for reading");

    
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (num_layers != bias.size()) {
        throw runtime_error("Network structure mismatch");
    }

    vector<size_t> layer_sizes(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (size != bias[i].size()) {
            throw runtime_error("Layer size mismatch");
        }
    }

    
    for (auto& layer : weights) {
        for (auto& neuron_weights : layer) {
            file.read(reinterpret_cast<char*>(neuron_weights.data()),
                      neuron_weights.size() * sizeof(double));
        }
    }

    for (auto& layer : bias) {
        file.read(reinterpret_cast<char*>(layer.data()),
                  layer.size() * sizeof(double));
    }
}