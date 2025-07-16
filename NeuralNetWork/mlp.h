#include <vector>
#include <iostream>
#include <random>


using namespace std;


class MLP {

    vector<vector<double>> bias;
    vector<vector<vector<double>>> weights;
    vector<vector<double>> data;    
    double alpha = 1.0;

    double relu(double x);

    double sigmoid(double x);

    double tanh_activation(double x);

    double swish(double x);

    std::vector<double> softmax(const std::vector<double>& z);

    double elu(double x);
    
    void calculate();
    
public:
    MLP(const vector<size_t>& neurons);

    vector<double>prdct(vector<double>&input);



};