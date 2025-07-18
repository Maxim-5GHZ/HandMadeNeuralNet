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


void MLP::validate_activations(const vector<string>& activations) {
    for (const auto& act : activations) {
        string lower_act = act;

        if (lower_act != "relu" && lower_act != "leaky_relu" &&
            lower_act != "sigmoid" && lower_act != "tanh" &&
            lower_act != "swish" && lower_act != "elu" &&
            lower_act != "silu" && lower_act != "gelu" &&
            lower_act != "selu" && lower_act != "softplus" &&
            lower_act != "softsign" && lower_act != "binary_step" &&
            lower_act != "identity" && lower_act != "softmax") {
            throw invalid_argument("Unknown activation function: " + act);
        }
    }
}


MLP::MLP(const vector<size_t>& neurons, const vector<string>& activations, double alpha)
    : activationNames(activations), alpha(alpha) {

    if (neurons.size() < 2) {
        throw invalid_argument("Network must have at least 2 layers (input and output)");
    }

    if (activations.size() != neurons.size() - 1) {
        throw invalid_argument("Number of activations must be equal to number of layers - 1");
    }

    validate_activations(activations);

  
    bias.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        bias[i].resize(neurons[i]);
        for (auto& val : bias[i]) {
            val = random_double(-0.1, 0.1);
        }
    }


    weights.resize(neurons.size() - 1);
    for (size_t i = 0; i < neurons.size() - 1; ++i) {
        double scale = sqrt(2.0 / neurons[i]);
        weights[i].resize(neurons[i]);
        for (size_t j = 0; j < neurons[i]; ++j) {
            weights[i][j].resize(neurons[i+1]);
            for (size_t k = 0; k < neurons[i+1]; ++k) {
                weights[i][j][k] = random_double(-scale, scale);
            }
        }
    }

   
    data.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        data[i].resize(neurons[i]);
    }
}


double MLP::relu(double x) { return max(0.0, x); }
double MLP::leaky_relu(double x) { return x > 0 ? x : leaky_relu_alpha * x; }
double MLP::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double MLP::tanh_activation(double x) { return tanh(x); }
double MLP::swish(double x) { return x * sigmoid(x); }
double MLP::elu(double x) { return x >= 0 ? x : alpha * (exp(x) - 1); }
double MLP::silu(double x) { return x * sigmoid(x); }
double MLP::gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
double MLP::selu(double x) {
    return selu_scale * (x > 0 ? x : selu_alpha * (exp(x) - 1));
}
double MLP::softplus(double x) { return log(1.0 + exp(x)); }
double MLP::softsign(double x) { return x / (1.0 + abs(x)); }
double MLP::binary_step(double x) { return x < 0 ? 0 : 1; }
double MLP::identity(double x) { return x; }

vector<double> MLP::softmax(const vector<double>& z) {
    vector<double> res(z.size());
    double max_z = *max_element(z.begin(), z.end());
    double sum_exp = 0.0;

    for (size_t i = 0; i < z.size(); ++i) {
        res[i] = exp(z[i] - max_z);
        sum_exp += res[i];
    }

    for (auto& val : res) val /= sum_exp;
    return res;
}

double MLP::relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }
double MLP::leaky_relu_derivative(double x) { return x > 0 ? 1.0 : leaky_relu_alpha; }
double MLP::sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}
double MLP::tanh_derivative(double x) {
    return 1.0 - tanh_activation(x) * tanh_activation(x);
}
double MLP::swish_derivative(double x) {
    double s = sigmoid(x);
    return s + x * s * (1 - s);
}
double MLP::elu_derivative(double x) { return x >= 0 ? 1.0 : alpha * exp(x); }
double MLP::silu_derivative(double x) {
    double s = sigmoid(x);
    return s + x * s * (1 - s);
}
double MLP::gelu_derivative(double x) {
    double cdf = 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
    return cdf + x * (1.0 / sqrt(2 * M_PI)) * exp(-0.5 * x * x) * (1 + 0.134145 * x * x);
}
double MLP::selu_derivative(double x) {
    return selu_scale * (x > 0 ? 1.0 : selu_alpha * exp(x));
}
double MLP::softplus_derivative(double x) { return 1.0 / (1.0 + exp(-x)); }
double MLP::softsign_derivative(double x) {
    double denom = 1.0 + abs(x);
    return 1.0 / (denom * denom);
}
double MLP::binary_step_derivative(double x) { return 0.0; }
double MLP::identity_derivative(double x) { return 1.0; }

vector<double> MLP::softmax_derivative(const vector<double>& z) {
    vector<double> sm = softmax(z);
    vector<double> derivative(z.size() * z.size(), 0.0);
    for (size_t i = 0; i < z.size(); ++i) {
        for (size_t j = 0; j < z.size(); ++j) {
            derivative[i * z.size() + j] = sm[i] * ((i == j) ? 1.0 - sm[j] : -sm[j]);
        }
    }
    return derivative;
}


void MLP::calculate() {
    for (size_t layer = 1; layer < data.size(); layer++) {
        
        for (size_t neuron = 0; neuron < data[layer].size(); neuron++) {
            double sum = bias[layer][neuron];
            for (size_t prev_neuron = 0; prev_neuron < data[layer-1].size(); prev_neuron++) {
                sum += weights[layer-1][prev_neuron][neuron] * data[layer-1][prev_neuron];
            }
            data[layer][neuron] = sum;
        }


        string activation = activationNames[layer-1];
        transform(activation.begin(), activation.end(), activation.begin(),
                 [](unsigned char c) { return tolower(c); });

        if (activation == "relu") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return relu(x); });
        }
        else if (activation == "leaky_relu") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return leaky_relu(x); });
        }
        else if (activation == "sigmoid") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return sigmoid(x); });
        }
        else if (activation == "tanh") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return tanh_activation(x); });
        }
        else if (activation == "swish") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return swish(x); });
        }
        else if (activation == "elu") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return elu(x); });
        }
        else if (activation == "silu") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return silu(x); });
        }
        else if (activation == "gelu") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return gelu(x); });
        }
        else if (activation == "selu") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return selu(x); });
        }
        else if (activation == "softplus") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return softplus(x); });
        }
        else if (activation == "softsign") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return softsign(x); });
        }
        else if (activation == "binary_step") {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return binary_step(x); });
        }
        else if (activation == "identity") {
            ;
        }
        else if (activation == "softmax") {
            vector<double> sm_output = softmax(data[layer]);
            data[layer] = sm_output;
        }
    }
}

vector<double> MLP::train(const vector<double>& input, const vector<double>& target, double learning_rate) {
    if (input.size() != data[0].size()) {
        throw invalid_argument("Input size doesn't match network input layer size");
    }
    if (target.size() != data.back().size()) {
        throw invalid_argument("Target size doesn't match network output layer size");
    }

 
    data[0] = input;
    calculate();


    vector<vector<double>> gradients(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        gradients[i].resize(data[i].size(), 0.0);
    }

    
    for (size_t i = 0; i < data.back().size(); ++i) {
        gradients.back()[i] = data.back()[i] - target[i];
    }


    for (size_t layer = data.size() - 1; layer > 0; --layer) {
        string activation = activationNames[layer - 1];
        transform(activation.begin(), activation.end(), activation.begin(),
                 [](unsigned char c) { return tolower(c); });

        for (size_t neuron = 0; neuron < data[layer].size(); ++neuron) {
          
            double derivative = 1.0;
            if (activation == "relu") {
                derivative = relu_derivative(data[layer][neuron]);
            }
            else if (activation == "leaky_relu") {
                derivative = leaky_relu_derivative(data[layer][neuron]);
            }
            else if (activation == "sigmoid") {
                derivative = sigmoid_derivative(data[layer][neuron]);
            }
            else if (activation == "tanh") {
                derivative = tanh_derivative(data[layer][neuron]);
            }
            else if (activation == "swish") {
                derivative = swish_derivative(data[layer][neuron]);
            }
            else if (activation == "elu") {
                derivative = elu_derivative(data[layer][neuron]);
            }
            else if (activation == "silu") {
                derivative = silu_derivative(data[layer][neuron]);
            }
            else if (activation == "gelu") {
                derivative = gelu_derivative(data[layer][neuron]);
            }
            else if (activation == "selu") {
                derivative = selu_derivative(data[layer][neuron]);
            }
            else if (activation == "softplus") {
                derivative = softplus_derivative(data[layer][neuron]);
            }
            else if (activation == "softsign") {
                derivative = softsign_derivative(data[layer][neuron]);
            }
            else if (activation == "binary_step") {
                derivative = binary_step_derivative(data[layer][neuron]);
            }
            else if (activation == "identity") {
                derivative = identity_derivative(data[layer][neuron]);
            }
            else if (activation == "softmax") {
                vector<double> sm_derivative = softmax_derivative(data[layer]);
                derivative = sm_derivative[neuron];
            }

            gradients[layer][neuron] *= derivative;

            
            for (size_t prev_neuron = 0; prev_neuron < data[layer - 1].size(); ++prev_neuron) {
                gradients[layer - 1][prev_neuron] += gradients[layer][neuron] *
                                                   weights[layer - 1][prev_neuron][neuron];
            }
        }
    }

    
    double max_grad = 1.0;
    for (auto& layer_grads : gradients) {
        for (auto& grad : layer_grads) {
            grad = max(-max_grad, min(max_grad, grad));
        }
    }

    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        for (size_t prev_neuron = 0; prev_neuron < weights[layer].size(); ++prev_neuron) {
            for (size_t neuron = 0; neuron < weights[layer][prev_neuron].size(); ++neuron) {
                weights[layer][prev_neuron][neuron] -= learning_rate *
                                                      gradients[layer + 1][neuron] *
                                                      data[layer][prev_neuron];
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
        throw invalid_argument("Input size doesn't match network input layer size");
    }
    data[0] = input;
    calculate();
    return data.back();
}


void MLP::save_weights(const string& filename) const {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file for saving weights");
    }


    size_t num_layers = bias.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    
    for (const auto& layer : bias) {
        size_t layer_size = layer.size();
        file.write(reinterpret_cast<const char*>(&layer_size), sizeof(layer_size));
    }


    for (const auto& act : activationNames) {
        size_t act_size = act.size();
        file.write(reinterpret_cast<const char*>(&act_size), sizeof(act_size));
        file.write(act.c_str(), act_size);
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
    if (!file.is_open()) {
        throw runtime_error("Cannot open file for loading weights");
    }

    
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (num_layers != bias.size()) {
        throw runtime_error("Number of layers does not match");
    }

    vector<size_t> layer_sizes(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        file.read(reinterpret_cast<char*>(&layer_sizes[i]), sizeof(size_t));
        if (layer_sizes[i] != bias[i].size()) {
            throw runtime_error("Layer size does not match");
        }
    }

    
    vector<string> saved_activations;
    for (size_t i = 0; i < num_layers - 1; ++i) {
        size_t act_size;
        file.read(reinterpret_cast<char*>(&act_size), sizeof(act_size));
        string act(act_size, ' ');
        file.read(&act[0], act_size);
        saved_activations.push_back(act);
    }

    if (saved_activations != activationNames) {
        throw runtime_error("Activation functions do not match");
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
