#ifndef MLP_ACTIVATORS_HPP
#define MLP_ACTIVATORS_HPP

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <functional>
#include <stdexcept>

class Activator {
public:
    enum Function {
        RELU = 1,
        LEAKY_RELU,
        SIGMOID,
        TANH,
        SWISH,
        ELU,
        GELU,
        SELU,
        SOFTPLUS,
        SOFTSIGN,
        BINARY_STEP,
        IDENTITY
    };

    struct ActivationPair {
        std::function<double(double)> function;
        std::function<double(double)> derivative;
    };

    Activator(const std::vector<Function>& functions, 
              double alpha_val = 0.01,
              double selu_alpha_val = 1.67326,
              double selu_scale_val = 1.0507)
        : alpha(alpha_val), selu_alpha(selu_alpha_val), selu_scale(selu_scale_val) 
    {
        for (auto func : functions) {
            switch (func) {
                case RELU:
                    activations.push_back(relu);
                    derivatives.push_back(relu_derivative);
                    break;
                case LEAKY_RELU:
                    activations.push_back([this](double x) { return leaky_relu(x); });
                    derivatives.push_back([this](double x) { return leaky_relu_derivative(x); });
                    break;
                case SIGMOID:
                    activations.push_back(sigmoid);
                    derivatives.push_back(sigmoid_derivative);
                    break;
                case TANH:
                    activations.push_back(tanh_activation);
                    derivatives.push_back(tanh_derivative);
                    break;
                case SWISH:
                    activations.push_back(swish);
                    derivatives.push_back(swish_derivative);
                    break;
                case ELU:
                    activations.push_back([this](double x) { return elu(x); });
                    derivatives.push_back([this](double x) { return elu_derivative(x); });
                    break;
                case GELU:
                    activations.push_back(gelu);
                    derivatives.push_back(gelu_derivative);
                    break;
                case SELU:
                    activations.push_back([this](double x) { return selu(x); });
                    derivatives.push_back([this](double x) { return selu_derivative(x); });
                    break;
                case SOFTPLUS:
                    activations.push_back(softplus);
                    derivatives.push_back(softplus_derivative);
                    break;
                case SOFTSIGN:
                    activations.push_back(softsign);
                    derivatives.push_back(softsign_derivative);
                    break;
                case BINARY_STEP:
                    activations.push_back(binary_step);
                    derivatives.push_back(binary_step_derivative);
                    break;
                case IDENTITY:
                    activations.push_back(identity);
                    derivatives.push_back(identity_derivative);
                    break;
                default:
                    throw std::invalid_argument("Unknown activation function");
            }
        }
    }

    const std::vector<std::function<double(double)>>& getActivations() const { return activations; }
    const std::vector<std::function<double(double)>>& getDerivatives() const { return derivatives; }

    static double relu(double x) { return x > 0 ? x : 0.0; }
    static double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

    double leaky_relu(double x) const { return x > 0 ? x : alpha * x; }
    double leaky_relu_derivative(double x) const { return x > 0 ? 1.0 : alpha; }

    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    static double tanh_activation(double x) { return std::tanh(x); }
    static double tanh_derivative(double x) {
        double t = tanh_activation(x);
        return 1.0 - t * t;
    }

    static double swish(double x) { return x * sigmoid(x); }
    static double swish_derivative(double x) {
        double s = sigmoid(x);
        return s + x * s * (1 - s);
    }

    double elu(double x) const { return x >= 0 ? x : alpha * (std::exp(x) - 1); }
    double elu_derivative(double x) const { return x >= 0 ? 1.0 : alpha * std::exp(x); }

    static double gelu(double x) {
        const double pi = 3.14159265358979323846;
        return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / pi) * 
               (x + 0.044715 * std::pow(x, 3))));
    }
    static double gelu_derivative(double x) {
        const double pi = 3.14159265358979323846;
        double cdf = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / pi) * 
               (x + 0.044715 * std::pow(x, 3))));
        return cdf + x * (1.0 / std::sqrt(2 * pi)) * 
               std::exp(-0.5 * x * x) * (1 + 0.134145 * x * x);
    }

    double selu(double x) const {
        return selu_scale * (x > 0 ? x : selu_alpha * (std::exp(x) - 1));
    }
    double selu_derivative(double x) const {
        return selu_scale * (x > 0 ? 1.0 : selu_alpha * std::exp(x));
    }

    static double softplus(double x) { return std::log(1.0 + std::exp(x)); }
    static double softplus_derivative(double x) { return 1.0 / (1.0 + std::exp(-x)); }

    static double softsign(double x) { return x / (1.0 + std::abs(x)); }
    static double softsign_derivative(double x) {
        double denom = 1.0 + std::abs(x);
        return 1.0 / (denom * denom);
    }

    static double binary_step(double x) { return x < 0 ? 0 : 1; }
    static double binary_step_derivative(double x) { return 0.0; }

    static double identity(double x) { return x; }
    static double identity_derivative(double x) { return 1.0; }

private:
    const double alpha;
    const double selu_alpha;
    const double selu_scale;
    
    std::vector<std::function<double(double)>> activations;
    std::vector<std::function<double(double)>> derivatives;
};

#endif