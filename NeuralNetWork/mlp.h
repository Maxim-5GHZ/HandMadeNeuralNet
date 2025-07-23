#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>
#include <cmath>
#include <functional>
#include "mlpActivators.hpp"

template<typename T>
class MLP {
private:
    std::vector<std::vector<T>> bias;
    std::vector<std::vector<std::vector<T>>> weights;
    std::vector<std::vector<T>> data;
    std::vector<std::function<T(T)>> activations;
    std::vector<std::function<T(T)>> activationDerivatives;

    T random_float(T min, T max);
    void calculate();

public:
    MLP(const std::vector<size_t>& neurons,
        const std::vector<typename Activator<T>::Function>& activate,
        T maxBiasValue);

    std::vector<T> train(const std::vector<T>& input,
                         const std::vector<T>& target,
                         T learning_rate = T(0.01));

    std::vector<T> predict(const std::vector<T>& input);

    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);
};

// Явное инстанцирование для float и double
extern template class MLP<float>;
extern template class MLP<double>;

#endif