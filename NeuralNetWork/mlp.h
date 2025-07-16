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

    double leaky_relu_alpha = 0.01;   // For Leaky ReLU
    
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
    
    void calculate();
    
public:

    MLP(const vector<size_t>& neurons, const vector<string>& activations,double alpha = 1.0);
    
    vector<double> train();

    vector<double> prdct(vector<double>& input);

};