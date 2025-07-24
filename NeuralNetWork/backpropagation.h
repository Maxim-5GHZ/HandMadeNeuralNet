#include "Perceptrone.h"
#include "mlpActivators.hpp"

template<typename T>
class Backpropagation : public Perceptrone<T> {
public:
    Backpropagation(const std::vector<size_t>& neurons,
        const std::vector<typename Activator<T>::Function>& activate,
        T maxBiasValue);
    
    std::vector<T> train(const std::vector<T>& input, const std::vector<T>& target, T learning_rate);
};