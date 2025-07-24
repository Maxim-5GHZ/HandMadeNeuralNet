#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>
#include <cmath>
#include <functional>
#include "mlpActivators.hpp"

template<typename T>
class Perceptrone {
protected:
    std::vector<std::vector<T>> bias;
    std::vector<std::vector<std::vector<T>>> weights;
    std::vector<std::vector<T>> data;
    std::vector<std::function<T(T)>> activations;
    std::vector<std::function<T(T)>> activationDerivatives;

    T random_float(T min, T max);
    void calculate();

public:
    Perceptrone(const std::vector<size_t>& neurons,
        const std::vector<typename Activator<T>::Function>& activate,
        T maxBiasValue);

    std::vector<T> predict(const std::vector<T>& input);
   
    const std::vector<std::vector<std::vector<T>>>& get_weights() const;
    const std::vector<std::vector<T>>& get_biases() const;
    
    void set_weights(const std::vector<std::vector<std::vector<T>>>& new_weights);
    void set_biases(const std::vector<std::vector<T>>& new_biases);
    
    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);
};

extern template class Perceptrone<float>;
extern template class Perceptrone<double>;