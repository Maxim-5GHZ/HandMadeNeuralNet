#ifndef MLP_ACTIVATORS_HPP
#define MLP_ACTIVATORS_HPP
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <functional>


class Activators {
public:
    
    static inline const short RELU = 1;
    static inline const short LEAKY_RELU = 2;
    static inline const short SIGMOID = 3;
    static inline const short TANH = 4;
    static inline const short SWISH = 5;
    static inline const short ELU = 6;
    static inline const short SILU = 7;
    static inline const short GELU = 8;
    static inline const short SELU = 9;
    static inline const short SOFTPLUS = 10;
    static inline const short SOFTSIGN = 11;
    static inline const short BINARY_STEP = 12;
    static inline const short IDENTITY = 13;

private:
    struct Pair {
        Pair(std::function<double(double)> activ, std::function<double(double)> derev)
            : activation(std::move(activ)), derevation(std::move(derev)) {}

        std::function<double(double)> activation;
        std::function<double(double)> derevation;
    };

    static  constexpr double alpha = 0.01;
    static const constexpr double leaky_relu_alpha = 0.01;
    static const constexpr double selu_scale = 1.0507;
    static const constexpr double selu_alpha = 1.67326;


    static double relu(double x) { return x > 0 ? x : 0.0; }
    static double leaky_relu(double x) { return x > 0 ? x : leaky_relu_alpha * x; }
    static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    static double tanh_activation(double x) { return tanh(x); }
    static double swish(double x) { return x * sigmoid(x); }
    static double elu(double x) { return x >= 0 ? x : alpha * (exp(x) - 1); }
    static double silu(double x) { return x * sigmoid(x); }
    static double gelu(double x) {
        return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
    }
    static double selu(double x) {
        return selu_scale * (x > 0 ? x : selu_alpha * (exp(x) - 1));
    }
    static double softplus(double x) { return log(1.0 + exp(x)); }
    static double softsign(double x) { return x / (1.0 + abs(x)); }
    static double binary_step(double x) { return x < 0 ? 0 : 1; }
    static double identity(double x) { return x; }

 
    static double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }
    static double leaky_relu_derivative(double x) { return x > 0 ? 1.0 : leaky_relu_alpha; }
    static double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }
    static double tanh_derivative(double x) {
        return 1.0 - tanh_activation(x) * tanh_activation(x);
    }
    static double swish_derivative(double x) {
        double s = sigmoid(x);
        return s + x * s * (1 - s);
    }
    static double elu_derivative(double x) { return x >= 0 ? 1.0 : alpha * exp(x); }
    static double silu_derivative(double x) {
        double s = sigmoid(x);
        return s + x * s * (1 - s);
    }
    static double gelu_derivative(double x) {
        double cdf = 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
        return cdf + x * (1.0 / sqrt(2 * M_PI)) * exp(-0.5 * x * x) * (1 + 0.134145 * x * x);
    }
    static double selu_derivative(double x) {
        return selu_scale * (x > 0 ? 1.0 : selu_alpha * exp(x));
    }
    static double softplus_derivative(double x) { return 1.0 / (1.0 + exp(-x)); }
    static double softsign_derivative(double x) {
        double denom = 1.0 + abs(x);
        return 1.0 / (denom * denom);
    }
    static double binary_step_derivative(double x) { return 0.0; }
    static double identity_derivative(double x) { return 1.0; }

         std::map<short, Pair> mapFunc = {
        {RELU, {relu, relu_derivative}},
        {LEAKY_RELU, {leaky_relu, leaky_relu_derivative}},
        {SIGMOID, {sigmoid, sigmoid_derivative}},
        {TANH, {tanh_activation, tanh_derivative}},
        {SWISH, {swish, swish_derivative}},
        {ELU, {elu, elu_derivative}},
        {SILU, {silu, silu_derivative}},
        {GELU, {gelu, gelu_derivative}},
        {SELU, {selu, selu_derivative}},
        {SOFTPLUS, {softplus, softplus_derivative}},
        {SOFTSIGN, {softsign, softsign_derivative}},
        {BINARY_STEP, {binary_step, binary_step_derivative}},
        {IDENTITY, {identity, identity_derivative}}
    };

public:
    std::vector<std::function<double(double)>> activation;
    std::vector<std::function<double(double)>> derivative;

    Activators(std::vector<short> activatorStrings,double a)
{
   a;
        for (const auto& name : activatorStrings) {
            auto it = mapFunc.find(name);
            if (it != mapFunc.end()) {
                activation.push_back(it->second.activation);
                derivative.push_back(it->second.derevation);
            }
        }
    }
};


#endif 