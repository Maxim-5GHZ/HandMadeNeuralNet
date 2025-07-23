#ifndef MLP_ACTIVATORS_HPP
#define MLP_ACTIVATORS_HPP

#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <stdexcept>

template<typename T>
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
        std::function<T(T)> function;
        std::function<T(T)> derivative;
    };

    Activator(const std::vector<Function>& functions, 
              T alpha_val = T(0.01),
              T selu_alpha_val = T(1.67326),
              T selu_scale_val = T(1.0507))
        : alpha(alpha_val), selu_alpha(selu_alpha_val), selu_scale(selu_scale_val) 
    {
        for (auto func : functions) {
            switch (func) {
                case RELU:
                    activations.push_back(relu);
                    derivatives.push_back(relu_derivative);
                    break;
                case LEAKY_RELU:
                    activations.push_back([this](T x) { return leaky_relu(x); });
                    derivatives.push_back([this](T x) { return leaky_relu_derivative(x); });
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
                    activations.push_back([this](T x) { return elu(x); });
                    derivatives.push_back([this](T x) { return elu_derivative(x); });
                    break;
                case GELU:
                    activations.push_back(gelu);
                    derivatives.push_back(gelu_derivative);
                    break;
                case SELU:
                    activations.push_back([this](T x) { return selu(x); });
                    derivatives.push_back([this](T x) { return selu_derivative(x); });
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

    const std::vector<std::function<T(T)>>& getActivations() const { return activations; }
    const std::vector<std::function<T(T)>>& getDerivatives() const { return derivatives; }

    static T relu(T x) { return x > 0 ? x : T(0); }
    static T relu_derivative(T x) { return x > 0 ? T(1) : T(0); }

    T leaky_relu(T x) const { return x > 0 ? x : alpha * x; }
    T leaky_relu_derivative(T x) const { return x > 0 ? T(1) : alpha; }

    static T sigmoid(T x) { return T(1) / (T(1) + std::exp(-x)); }
    static T sigmoid_derivative(T x) {
        T s = sigmoid(x);
        return s * (T(1) - s);
    }

    static T tanh_activation(T x) { return std::tanh(x); }
    static T tanh_derivative(T x) {
        T t = tanh_activation(x);
        return T(1) - t * t;
    }

    static T swish(T x) { return x * sigmoid(x); }
    static T swish_derivative(T x) {
        T s = sigmoid(x);
        return s + x * s * (T(1) - s);
    }

    T elu(T x) const { return x >= 0 ? x : alpha * (std::exp(x) - T(1)); }
    T elu_derivative(T x) const { return x >= 0 ? T(1) : alpha * std::exp(x); }

    static T gelu(T x) {
        const T pi = T(3.14159265358979323846);
        return T(0.5) * x * (T(1) + std::tanh(std::sqrt(T(2) / pi) * 
               (x + T(0.044715) * std::pow(x, T(3)))));
    }
    static T gelu_derivative(T x) {
        const T pi = T(3.14159265358979323846);
        T cdf = T(0.5) * (T(1) + std::tanh(std::sqrt(T(2) / pi) * 
               (x + T(0.044715) * std::pow(x, T(3)))));
        return cdf + x * (T(1) / std::sqrt(T(2) * pi)) * 
               std::exp(T(-0.5) * x * x) * (T(1) + T(0.134145) * x * x);
    }

    T selu(T x) const {
        return selu_scale * (x > 0 ? x : selu_alpha * (std::exp(x) - T(1)));
    }
    T selu_derivative(T x) const {
        return selu_scale * (x > 0 ? T(1) : selu_alpha * std::exp(x));
    }

    static T softplus(T x) { return std::log(T(1) + std::exp(x)); }
    static T softplus_derivative(T x) { return T(1) / (T(1) + std::exp(-x)); }

    static T softsign(T x) { return x / (T(1) + std::abs(x)); }
    static T softsign_derivative(T x) {
        T denom = T(1) + std::abs(x);
        return T(1) / (denom * denom);
    }

    static T binary_step(T x) { return x < 0 ? T(0) : T(1); }
    static T binary_step_derivative(T x) { return T(0); }

    static T identity(T x) { return x; }
    static T identity_derivative(T x) { return T(1); }

private:
    const T alpha;
    const T selu_alpha;
    const T selu_scale;
    
    std::vector<std::function<T(T)>> activations;
    std::vector<std::function<T(T)>> derivatives;
};

#endif