#include "mlp.h"
#include <algorithm>
#include<random>
#include <iostream>

double random_double(double min, double max) {

    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dist(min, max);
    return dist(gen);

}

double MLP::relu(double x) { 
    return x > 0 ? x : 0; 
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
    return x > 0 ? x : alpha * (exp(x) - 1);
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
    for (size_t i = 1; i < data.size(); i++) {
       
        for (size_t j = 0; j < data[i].size(); j++) {
            double sum = bias[i][j];
            for (size_t k = 0; k < data[i-1].size(); k++) {
                sum += weights[i-1][k][j] * data[i-1][k];
            }
            data[i][j] = sum; 
        }

        string act = activationNames[i-1];
        transform(act.begin(), act.end(), act.begin(), 
                 [](unsigned char c) { return tolower(c); });

        if (act == "relu") {
            for (auto& val : data[i]) val = relu(val);
        } 
        else if (act == "sigmoid") {
            for (auto& val : data[i]) val = sigmoid(val);
        } 
        else if (act == "tanh") {
            for (auto& val : data[i]) val = tanh_activation(val);
        } 
        else if (act == "swish") {
            for (auto& val : data[i]) val = swish(val);
        } 
        else if (act == "elu") {
            for (auto& val : data[i]) val = elu(val);
        } 
        else if (act == "softmax") {
            vector<double> sm_output = softmax(data[i]);
            for (size_t j = 0; j < data[i].size(); j++) {
                data[i][j] = sm_output[j];
            }
        } 
        else {
            for (auto& val : data[i]) val = relu(val);
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

vector<double> MLP::prdct(vector<double>& input) {
    data[0] = input; 
    calculate();
    return data.back();
}