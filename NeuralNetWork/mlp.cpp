#include "mlp.h"
#include <algorithm>
#include<random>
#include <iostream>

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
    
    if (activations.size() != neurons.size() - 1) {
        throw invalid_argument("Err");
    }

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

vector<double> train(){
    
}

vector<double> MLP::prdct(vector<double>& input) {
    data[0] = input; 
    calculate();
    return data.back();
}


