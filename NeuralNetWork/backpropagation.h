#include "Perceptrone.h"
#include "mlpActivators.hpp"

template<typename T>
class Backpropagation : public Perceptrone<T> {
public:
    Backpropagation(Perceptrone<T>&perceptrone);
    
    std::vector<T> train(const std::vector<T>& input, const std::vector<T>& target, T learning_rate);
};