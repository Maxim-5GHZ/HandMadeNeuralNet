#include "Perceptrone.h"

template<typename T>
  
class Genetic{
    
    struct Gen{
        T fitnessk;
        std::vector<Perceptrone<T>> model;
    };

    std::vector<<Perceptrone<T>> selectModel

    std::vector<Gen> generations;

    void mutate();

public:
    Genetic(const std::vector<size_t>& neurons,
        const std::vector<typename Activator<T>::Function>& activate,
        T maxBiasValue,size_t populationSize);
    
    Perceptrone<T> getModel(size_t numModel,size_t numGen);
    void setFitness(size_t numModel,size_t numGen,T fitness);
    
    void tourSelect();
    void rouletteSelect();

    
};  