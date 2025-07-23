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
        std::function<float(float)> function;
        std::function<float(float)> derivative;
    };

    Activator(const std::vector<Function>& functions, 
              float alpha_val = 0.01,
              float selu_alpha_val = 1.67326,
              float selu_scale_val = 1.0507)
        : alpha(alpha_val), selu_alpha(selu_alpha_val), selu_scale(selu_scale_val) 
    {
        for (auto func : functions) {
            switch (func) {
                case RELU:
                    activations.push_back(relu);
                    derivatives.push_back(relu_derivative);
                    break;
                case LEAKY_RELU:
                    activations.push_back([this](float x) { return leaky_relu(x); });
                    derivatives.push_back([this](float x) { return leaky_relu_derivative(x); });
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
                    activations.push_back([this](float x) { return elu(x); });
                    derivatives.push_back([this](float x) { return elu_derivative(x); });
                    break;
                case GELU:
                    activations.push_back(gelu);
                    derivatives.push_back(gelu_derivative);
                    break;
                case SELU:
                    activations.push_back([this](float x) { return selu(x); });
                    derivatives.push_back([this](float x) { return selu_derivative(x); });
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

    const std::vector<std::function<float(float)>>& getActivations() const { return activations; }
    const std::vector<std::function<float(float)>>& getDerivatives() const { return derivatives; }

    static float relu(float x) { return x > 0 ? x : 0.0; }
    static float relu_derivative(float x) { return x > 0 ? 1.0 : 0.0; }

    float leaky_relu(float x) const { return x > 0 ? x : alpha * x; }
    float leaky_relu_derivative(float x) const { return x > 0 ? 1.0 : alpha; }

    static float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }
    static float sigmoid_derivative(float x) {
        float s = sigmoid(x);
        return s * (1 - s);
    }

    static float tanh_activation(float x) { return std::tanh(x); }
    static float tanh_derivative(float x) {
        float t = tanh_activation(x);
        return 1.0 - t * t;
    }

    static float swish(float x) { return x * sigmoid(x); }
    static float swish_derivative(float x) {
        float s = sigmoid(x);
        return s + x * s * (1 - s);
    }

    float elu(float x) const { return x >= 0 ? x : alpha * (std::exp(x) - 1); }
    float elu_derivative(float x) const { return x >= 0 ? 1.0 : alpha * std::exp(x); }

    static float gelu(float x) {
        const float pi = 3.14159265358979323846;
        return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / pi) * 
               (x + 0.044715 * std::pow(x, 3))));
    }
    static float gelu_derivative(float x) {
        const float pi = 3.14159265358979323846;
        float cdf = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / pi) * 
               (x + 0.044715 * std::pow(x, 3))));
        return cdf + x * (1.0 / std::sqrt(2 * pi)) * 
               std::exp(-0.5 * x * x) * (1 + 0.134145 * x * x);
    }

    float selu(float x) const {
        return selu_scale * (x > 0 ? x : selu_alpha * (std::exp(x) - 1));
    }
    float selu_derivative(float x) const {
        return selu_scale * (x > 0 ? 1.0 : selu_alpha * std::exp(x));
    }

    static float softplus(float x) { return std::log(1.0 + std::exp(x)); }
    static float softplus_derivative(float x) { return 1.0 / (1.0 + std::exp(-x)); }

    static float softsign(float x) { return x / (1.0 + std::abs(x)); }
    static float softsign_derivative(float x) {
        float denom = 1.0 + std::abs(x);
        return 1.0 / (denom * denom);
    }

    static float binary_step(float x) { return x < 0 ? 0 : 1; }
    static float binary_step_derivative(float x) { return 0.0; }

    static float identity(float x) { return x; }
    static float identity_derivative(float x) { return 1.0; }

private:
    const float alpha;
    const float selu_alpha;
    const float selu_scale;
    
    std::vector<std::function<float(float)>> activations;
    std::vector<std::function<float(float)>> derivatives;
};

#endif