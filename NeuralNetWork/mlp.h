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

    double relu(double x);

    double sigmoid(double x);

    double tanh_activation(double x);

    double swish(double x);

    vector<double> softmax(const vector<double>& z); 
    
    double elu(double x);
    
    void calculate();
    
public:

    MLP(const vector<size_t>& neurons, const vector<string>& activations,double alpha = 1.0);
    
    vector<double> train();

    vector<double> prdct(vector<double>& input);

};