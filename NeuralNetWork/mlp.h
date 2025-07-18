#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

class MLP {
private:
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> data;
    std::vector<std::string> activationNames;


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
    std::vector<double> softmax(const std::vector<double>& z);

   
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
    std::vector<double> softmax_derivative(const std::vector<double>& z);

   
    void calculate();
    void validate_activations(const std::vector<std::string>& activations);
    double random_double(double min, double max);

public:
    
    MLP(const std::vector<size_t>& neurons,
        const std::vector<std::string>& activations,
        double alpha = 1.0);

   
    std::vector<double> train(const std::vector<double>& input,
                             const std::vector<double>& target,
                             double learning_rate = 0.01);
    std::vector<double> predict(const std::vector<double>& input);


    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);
};

#endif // MLP_H
