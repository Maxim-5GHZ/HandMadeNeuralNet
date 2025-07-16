#include "mlp.h"
#include <algorithm>
#include<random>

double random_double(double min = -1, double max = 1) {

    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dist(min, max);
    return dist(gen);

}

double MLP::relu(double x) { return max(0.0, x); }

double MLP::leaky_relu(double x) { 
    return x > 0 ? x : leaky_relu_alpha * x; 
}

double MLP::sigmoid(double x) { 
    return 1.0 / (1.0 + exp(-x)); 
}

double MLP::tanh_activation(double x) { 
    return tanh(x); 
}

double MLP::swish(double x) { 
    return x * sigmoid(x); 
}

double MLP::elu(double x) { 
    return x >= 0 ? x : alpha * (exp(x) - 1); 
}

double MLP::silu(double x) { 
    return x * sigmoid(x); 
}

double MLP::gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

double MLP::selu(double x) {
    return selu_scale * (x > 0 ? x : selu_alpha * (exp(x) - 1));
}

double MLP::softplus(double x) {
    return log(1.0 + exp(x));
}

double MLP::softsign(double x) {
    return x / (1.0 + abs(x));
}

double MLP::binary_step(double x) {
    return x < 0 ? 0 : 1;
}

double MLP::identity(double x) {
    return x;
}

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

double MLP::leaky_relu_derivative(double x) { 
    return x > 0 ? 1.0 : leaky_relu_alpha; 
}

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

double MLP::elu_derivative(double x) { 
    return x >= 0 ? 1.0 : alpha * exp(x); 
}

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

double MLP::softplus_derivative(double x) { 
    return 1.0 / (1.0 + exp(-x)); 
}

double MLP::softsign_derivative(double x) { 
    double denom = 1.0 + abs(x);
    return 1.0 / (denom * denom); 
}

double MLP::binary_step_derivative(double x) { 
    return 0.0; 
}

double MLP::identity_derivative(double x) { 
    return 1.0; 
}

vector<double> MLP::softmax_derivative(const vector<double>& z) {
    vector<double> sm = softmax(z);
    vector<double> derivative(z.size(), 0.0);
    for (size_t i = 0; i < z.size(); ++i) {
        derivative[i] = sm[i] * (1 - sm[i]);
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
            
        }
        else if (activation == "softmax") {
            vector<double> sm_output = softmax(data[layer]);
            data[layer] = sm_output;
        }
        else {
            transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                     [this](double x) { return relu(x); });
        }
    }
}


MLP::MLP(const vector<size_t>& neurons, const vector<string>& activations,double alpha) 
    : activationNames(activations) {
    
    this->alpha = alpha;
    
   
    size_t layers = neurons.size();

    bias.resize(layers);
    for (size_t i = 0; i < layers; ++i) {
        bias[i].resize(neurons[i]);
        for (auto& val : bias[i]) val = random_double(-0.5, 0.5);
    }

   
    weights.resize(layers - 1);
    for (size_t i = 0; i < layers - 1; ++i) {
        weights[i].resize(neurons[i]);
        for (size_t j = 0; j < neurons[i]; ++j) {
            weights[i][j].resize(neurons[i+1]);
            for (size_t k = 0; k < neurons[i+1]; ++k) {
                weights[i][j][k] = random_double(-1, 1);
            }
        }
    }


    data.resize(layers);
    for (size_t i = 0; i < layers; ++i) {
        data[i].resize(neurons[i]);
    }
}

vector<double> MLP::train(vector<double>& input, vector<double>& target, double learning_rate) {
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
                gradients[layer - 1][prev_neuron] += gradients[layer][neuron] * weights[layer - 1][prev_neuron][neuron];
            }
        }
    }

    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        for (size_t prev_neuron = 0; prev_neuron < weights[layer].size(); ++prev_neuron) {
            for (size_t neuron = 0; neuron < weights[layer][prev_neuron].size(); ++neuron) {
                weights[layer][prev_neuron][neuron] -= learning_rate * gradients[layer + 1][neuron] * data[layer][prev_neuron];
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

vector<double> MLP::prdct(vector<double>& input) {
    data[0] = input; 
    calculate();
    return data.back();
}


