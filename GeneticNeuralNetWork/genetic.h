#ifndef GENETIC_H
#define GENETIC_H

#include "Perceptrone.h"
#include <vector>
#include <random>

template<typename T>
class Genetic {
    struct Gen {
        T fitness;
        Perceptrone<T> model;
        
        Gen(const Perceptrone<T>& m, T f = T(0)) : model(m), fitness(f) {}
    };

    std::vector<Gen> generations;
    std::mt19937 gen;

    void mutate(Gen& gen, T mutationRate);

public:
    Genetic(const std::vector<size_t>& neurons,
            const std::vector<typename Activator<T>::Function>& activate,
            T maxBiasValue, size_t populationSize);
    
    Perceptrone<T>& getModel(size_t numModel);
    const Perceptrone<T>& getModel(size_t numModel) const;
    void setFitness(size_t numModel, T fitness);
    
    void tourSelect(size_t tournamentSize);
    void rouletteSelect();
    void mutate(size_t index, T mutationRate);
};

#endif