#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>
#include "mlpActivators.hpp"

class MLP {
private:
    std::vector<std::vector<float>> bias;
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> data;
    std::vector<std::function<float(float)>> activations;
    std::vector<std::function<float(float)>> activationDerivatives;

    float random_float(float min, float max);
    void calculate();

public:
    MLP(const std::vector<size_t>& neurons,
        const std::vector<Activator::Function>& activate,
        float maxBiasValue);

    std::vector<float> train(const std::vector<float>& input,
                             const std::vector<float>& target,
                             float learning_rate = 0.01);

    std::vector<float> predict(const std::vector<float>& input);

    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);
};

#endif