#include <vector>
#include <string>
#pragma once


using namespace std;

class MLP {

    vector<vector<double>> bias;

    vector<vector<vector<double>>> weights;

    vector<vector<double>> data;
    
    vector<string> activationNames; 
    
    double alpha = 1.0;

    double leaky_relu_alpha = 0.01;   
    
    const double selu_scale = 1.0507;
    
    const double selu_alpha = 1.67326;

    double relu(double x);
    
    double leaky_relu(double x);
    
    double sigmoid(double x);
    
    double tanh_activation(double x);
    
    double swish(double x);
    
    double elu(double x);
    
    double silu(double x);
    
    double gelu(double x);
    
    double selu(double x);
    
    double softplus(double x);
    
    double softsign(double x);
    
    double binary_step(double x);
    
    double identity(double x);
    
    vector<double> softmax(const vector<double>& z);

    double relu_derivative(double x);

    double leaky_relu_derivative(double x);
    
    double sigmoid_derivative(double x);
    
    double tanh_derivative(double x);
    
    double swish_derivative(double x);
    
    double elu_derivative(double x);
    
    double silu_derivative(double x);
    
    double gelu_derivative(double x);
    
    double selu_derivative(double x);
    
    double softplus_derivative(double x);
    
    double softsign_derivative(double x);
    
    double binary_step_derivative(double x);
    
    double identity_derivative(double x);
    
    vector<double> softmax_derivative(const vector<double>& z);
    
    void calculate();
    
public:

    MLP(const vector<size_t>& neurons, const vector<string>& activations,double alpha = 1.0);
    
    vector<double> train(vector<double>& input, vector<double>& target, double learning_rate = 0.01);

    vector<double> prdct(vector<double>& input);

};